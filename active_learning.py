import argparse
from utils import *
from training_utils import *
from torch.utils.data.dataset import *
from torch.utils.data.sampler import *
from torch.nn.utils.rnn import *
import bisect
from model import *
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


class ActiveLearningDataset(object):
    def __init__(
        self,
        train_data,
        test_data,
        budget,
        round_size,
        batch_size,
        drop_last=False,
        device="cuda",
    ):
        """
        train_data: loaded from pickle
        test_data: loaded from pickle

        word_scoring_func: function used to score single words
        Inputs:
            output: Tensor, shape (batch size, sequence length, number of possible tags), model outputs of all instances
            targets: Tensor, shape (batch size, sequence length), ground truth, indexing output
        Outputs:
            a score, with higher meaning better to pick

        budget: total number of elements we can label (words)
        round_size: total number instances we label each round (sentences)
        """

        # !!! Right now, following the pipeline, we assume we can do word-wise aggregation of scores
        # This might have to change....
        self.word_scoring_func = nn.NLLLoss(reduce=False)
        # True means we do 1-mean(word scores) like with MLP, False means we do mean(word scores) like with BALD
        self.inverse_score_flag = True

        self.score_extraction = self.full_sentence_extraction  # The baseline

        self.budget = budget
        self.round_size = round_size
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.train_data = train_data
        self.test_data = test_data

        # Dictionaries mapping {sentence idx: [list, of, word, idx]} for labelled and unlabelled words
        self.labelled_idx = {j: [] for j in range(len(self.train_data))}
        self.unlabelled_idx = {
            j: list(range(len(train_data[j][0]))) for j in range(len(self.train_data))
        }
        self.update_datasets()

        self.device = device

    @staticmethod
    def full_sentence_extraction(scores_list):
        """
        Input:
            scores_list: [list, of, scores, from, a, sentence, None, None]
            None represents previously labelled word - which will not appear for this strategy
        Output:
            entries = [([list, of, word, idx], score), ...] for all possible extraction batches
            For this strategy, entries is one element, with all the indices of this sentence
        """
        score = mean(scores_list)
        indices = list(range(len(scores_list)))

        return [(score, indices)]

    def random_init(self):
        """
        TODO
        Randomly initialise self.labelled_idx dictionary
        """

    def scores_from_batch_idx(self, model, batch_index, train_data=None, device=None):
        """
        Input:
            batch_index: list of one integer

        Outputs:
            tensor of size [1, sequence length] showing per word score
        """

        if not train_data:
            train_data = self.train_data
        if not device:
            device = self.device

        # Will have to change this to extend to BALD, as it requires multiple model passes
        sentences, tokens, targets, lengths = get_batch(batch_index, train_data, device)
        output = model(sentences, tokens)
        output = pack_padded_sequence(output, lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, lengths, batch_first=True).data
        word_scores = self.word_scoring_func(
            output, targets
        )  # Without per-word reduction, i.e. a list of scores of size sentence size
        return word_scores

    def extend_indices(self, sentence_scores):
        """
        After a full pass on the unlabelled pool, apply a policy to get the top scoring phrases.
        ! THIS INCLUDES previously labelled words, i.e. it becomes the new

        Input:
            sentence_scores: {j: [list, of, scores, per, word, None, None]} where None means the word has alread been labelled
                                                                                i.e. full list of scores/Nones
        Output:
            No output, but extends self.labelled_idx:
            {
                j: [5, 6, 7],
                i: [1, 2, 3, 8, 9, 10, 11],
                ...
            }
            meaning words 5, 6, 7 of word j are chosen to be labelled.
            !! This includes previously labelled words, so extends the full dataset partition
        """
        temp_score_list = [
            (-1, [], -np.inf) for _ in range(self.round_size)
        ]  # (sentence_idx, [list, of, word, idx], score) to be added to self.labelled_idx at the end
        scores_from_temp_dict = lambda x=temp_score_list: [
            a[-1] for a in x
        ]  # Get list of scores

        for sentence_idx, scores_list in sentence_scores.items():
            entries = self.score_extraction(
                scores_list
            )  # [([list, of, word, idx], score), ...] that can be compared to temp_score_list
            for entry in entries:
                if entry[-1] > min(scores_from_temp_dict()):
                    temp_score_list[0] = [sentence_idx, entry]
                    temp_score_list.sort(key=lambda y: y[-1])
                else:
                    pass

        for sentence_idx, word_inds, score in temp_score_list:
            self.labelled_idx[sentence_idx].extend(word_inds)

        self.unlabelled_idx = {
            i: [j for j in self.unlabelled_idx if j not in self.labelled_idx[i]]
            for i in self.unlabelled_idx
        }

    def update_indices(self, model):
        """
        Score unlabelled instances in terms of their suitability to be labelled next.
        Add the highest scoring instance indices in the dataset to self.labelled_idx
        """
        sentence_scores = {}
        for idx, batch_index in enumerate(self.unlabelled_batch_indices):

            word_scores = self.scores_from_batch_idx(
                model, batch_index
            )  # Get raw score tensor for that sentence
            b = batch_index[0]
            sentence_scores[b] = [
                int(score[j]) if j in self.unlabelled_idx[b] else None
                for j in range(len(score))
            ]  # scores of unlabelled words --> int, scores of labelled words --> None

        self.extend_indices(sentence_scores)

    def update_datasets(self):
        """
        After ranking the full dataset, use the extended self.labelled_idx to create
        new dataset objects for labelled and unlabelled instances
        """

        # TODO: We might want to change the threshold number labels needed to include sentence
        # Right now it is just one (i.e. not empty)
        partially_labelled_sentence_idx = [
            j for j in self.labelled_idx.keys() if self.labelled_idx[j]
        ]
        labelled_subset = Subset(self.train_data, partially_labelled_sentence_idx)

        self.labelled_batch_indices = list(
            BatchSampler(
                SubsetRandomSampler(labelled_subset.indices),
                self.batch_size,
                drop_last=self.drop_last,
            )
        )
    
    def __iter__(self):
        # TODO: Write get_batch_active which decides which ones to hand/auto-label
        return (
            self.labelled_batch_indices[i]
            for i in torch.randperm(len(self.labelled_batch_indices))
        )

    def __len__(self):
        return len(self.labelled_batch_indices)


# from train import *; from active_learning import ActiveLearningDataset; a = ActiveLearningDataset(train_data, test_data, 5, 6000, args.batch_size)