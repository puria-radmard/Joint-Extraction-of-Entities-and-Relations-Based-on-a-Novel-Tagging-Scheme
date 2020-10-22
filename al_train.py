from training_utils import *
from active_learning import ActiveLearningDataset, FullSentenceLowestConfidence
from tqdm import tqdm

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

parser = argparse.ArgumentParser(
    description="Joint Extraction of Entities and Relations"
)
parser.add_argument(
    "--batch_size", type=int, default=32, metavar="N", help="batch size (default: 32)"
)
parser.add_argument("--cuda", action="store_false", help="use CUDA (default: True)")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    help="dropout applied to layers (default: 0.5)",
)
parser.add_argument(
    "--emb_dropout",
    type=float,
    default=0.25,
    help="dropout applied to the embedded layer (default: 0.25)",
)
parser.add_argument(
    "--clip",
    type=float,
    default=0.35,
    help="gradient clip, -1 means no clip (default: 0.35)",
)
# debug
parser.add_argument(
    "--epochs", type=int, default=30, help="upper epoch limit (default: 30)"
)
parser.add_argument(
    "--char_kernel_size",
    type=int,
    default=3,
    help="character-level kernel size (default: 3)",
)
parser.add_argument(
    "--word_kernel_size",
    type=int,
    default=3,
    help="word-level kernel size (default: 3)",
)
parser.add_argument(
    "--emsize", type=int, default=50, help="size of character embeddings (default: 50)"
)
parser.add_argument(
    "--char_layers",
    type=int,
    default=3,
    help="# of character-level convolution layers (default: 3)",
)
parser.add_argument(
    "--word_layers",
    type=int,
    default=3,
    help="# of word-level convolution layers (default: 3)",
)
parser.add_argument(
    "--char_nhid",
    type=int,
    default=50,
    help="number of hidden units per character-level convolution layer (default: 50)",
)
parser.add_argument(
    "--word_nhid",
    type=int,
    default=300,
    help="number of hidden units per word-level convolution layer (default: 300)",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="report interval (default: 100)",
)
parser.add_argument(
    "--lr", type=float, default=4, help="initial learning rate (default: 4)"
)
parser.add_argument(
    "--optim", type=str, default="SGD", help="optimizer type (default: SGD)"
)
parser.add_argument(
    "--seed", type=int, default=1111, help="random seed (default: 1111)"
)
parser.add_argument(
    "--save", type=str, default="model.pt", help="path to save the final model"
)
parser.add_argument(
    "--weight",
    type=float,
    default=10.0,
    help='manual rescaling weight given to each tag except "O"',
)
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

char_channels = [args.emsize] + [args.char_nhid] * args.char_layers
word_channels = [word_embedding_size + args.char_nhid] + [
    args.word_nhid
] * args.word_layers


def make_model():
    model = Model(
        charset_size=len(charset),
        char_embedding_size=args.emsize,
        char_channels=char_channels,
        char_padding_idx=charset["<pad>"],
        char_kernel_size=args.char_kernel_size,
        weight=word_embeddings,
        word_embedding_size=word_embedding_size,
        word_channels=word_channels,
        word_kernel_size=args.word_kernel_size,
        num_tag=len(tag_set),
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
    ).to(device)
    return model


weight = [args.weight] * len(tag_set)
weight[tag_set["O"]] = 1
weight = torch.tensor(weight).to(device)
criterion = nn.NLLLoss(weight, size_average=False)

def early_stopping_original(f1_list, num):
    if len(f1_list) < num:
        return False
    elif sorted(f1_list[-num:], reverse = True) == f1_list[-num:]:
        return True
    else:
        return False

def early_stopping(f1_list, num=3):
    if len(f1_list) < num:
        return False
    elif len(f1_list)-np.argmax(f1_list) > num:
        return True
    else:
        return False


def train(model, al_agent):

    model.train()
    total_loss = 0
    count = 0
    sampler = al_agent.labelled_batch_indices

    for idx, batch_indices in enumerate(sampler):
        sentences, tokens, targets, lengths = get_batch(
            batch_indices, al_agent.autolabelled_data, device
        )

        # [vocab[int(a)] for a in sentences[10]] reveals sentence
        # [tag_set[int(a)] for a in targets[10]] reveals corresponding taglist

        optimizer.zero_grad()
        output = model(sentences, tokens)
        # output in shape [batch_size, length_of_sentence, num_tags (193)]

        output = pack_padded_sequence(output, lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, lengths, batch_first=True).data
        loss = criterion(output, targets)
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        count += len(targets)

        if (idx + 1) % args.log_interval == 0:
            cur_loss = total_loss / count
            elapsed = time.time() - start_time
            percent = ((epoch - 1) * len(sampler) + (idx + 1)) / (
                args.epochs * len(sampler)
            )
            remaining = elapsed / percent - elapsed
            print(
                "| Epoch {:2d}/{:2d} | Batch {:5d}/{:5d} | Elapsed Time {:s} | Remaining Time {:s} | "
                "lr {:4.2e} | Loss {:5.3f} |".format(
                    epoch,
                    args.epochs,
                    idx + 1,
                    len(sampler),
                    time_display(elapsed),
                    time_display(remaining),
                    lr,
                    cur_loss,
                )
            )
            total_loss = 0
            count = 0


# NOT CHANGED BY AL
def evaluate(model, data_groups):
    model.eval()
    total_loss = 0
    count = 0
    TP = 0
    TP_FP = 0
    TP_FN = 0
    with torch.no_grad():
        print("Beginning evaluation")
        for batch_indices in tqdm(
            GroupBatchRandomSampler(data_groups, args.batch_size, drop_last=False)
        ):
            sentences, tokens, targets, lengths = get_batch(
                batch_indices, train_data, device
            )

            output = model(sentences, tokens)
            tp, tp_fp, tp_fn = measure(output, targets, lengths)
            # tp, tp_fp, tp_fn = 5,5,5

            TP += tp
            TP_FP += tp_fp
            TP_FN += tp_fn
            output = pack_padded_sequence(output, lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, lengths, batch_first=True).data
            loss = criterion(output, targets)
            total_loss += loss.item()

            count += len(targets)
    if TP_FP == 0:
        TP_FP = 1
    if TP_FN == 0:
        TP_FN = 1
    return total_loss / count, TP / TP_FP, TP / TP_FN, 2 * TP / (TP_FP + TP_FN)


def measure(output, targets, lengths):
    assert output.size(0) == targets.size(0) and targets.size(0) == lengths.size(0)
    tp = 0
    tp_fp = 0
    tp_fn = 0
    batch_size = output.size(0)
    output = torch.argmax(output, dim=-1)
    for i in range(batch_size):
        length = lengths[i]
        out = output[i][:length].tolist()
        target = targets[i][:length].tolist()
        out_triplets = get_triplets(out)
        tp_fp += len(out_triplets)
        target_triplets = get_triplets(target)
        tp_fn += len(target_triplets)
        for target_triplet in target_triplets:
            for out_triplet in out_triplets:
                if out_triplet == target_triplet:
                    tp += 1
    return tp, tp_fp, tp_fn


if __name__ == "__main__":

    model = make_model()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

    agent = FullSentenceLowestConfidence(
        train_data=train_data,
        test_data=test_data,
        num_sentences_init=int(len(train_data) * 0.1),
        round_size=int(len(train_data) * 0.05),
        batch_size=args.batch_size,
        model=model,
    )

    while agent.budget > 0:

        best_val_loss = None
        lr = args.lr
        all_val_loss = []
        all_precision = []
        all_recall = []
        all_f1 = []

        start_time = time.time()
        print("-" * 118)

        word_inds = [v for k, v in agent.labelled_idx.items() if v]
        num_sentences = len(word_inds)
        num_words = sum([len(b) for b in word_inds])

        print(
            f"Starting training with {num_words} words labelled in {num_sentences} sentences"
        )
        for epoch in range(1, args.epochs + 1):

            if early_stopping(all_f1, 5):
                break
            
            train(model=model, al_agent=agent)

            val_loss, precision, recall, f1 = evaluate(model, val_data_groups)

            elapsed = time.time() - start_time
            print(
                "| End of Epoch {:2d} | Elapsed Time {:s} | Validation Loss {:5.3f} | Precision {:5.3f} "
                "| Recall {:5.3f} | F1 {:5.3f} |".format(
                    epoch, time_display(elapsed), val_loss, precision, recall, f1
                )
            )

                # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr = lr / 4.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            all_val_loss.append(val_loss)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

        # Run on test data
        test_loss, precision, recall, f1 = evaluate(model, test_data_groups)
        print("=" * 118)
        print(
            "| End of Training | Test Loss {:5.3f} | Precision {:5.3f} "
            "| Recall {:5.3f} | F1 {:5.3f} |".format(test_loss, precision, recall, f1)
        )
        print("=" * 118)

        with open(f"record-{num_sentences}.tsv", "wt", encoding="utf-8") as f:
            f.write(
                f"{num_words} words in {num_sentences} sentences. Total {len(train_data)} sentences.\n\n"
            )
            f.write("Epoch\tLOSS\tPREC\tRECL\tF1\n")
            for idx in range(len(all_val_loss)):
                f.write(
                    "{:d}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".format(
                        idx + 1,
                        all_val_loss[idx],
                        all_precision[idx],
                        all_recall[idx],
                        all_f1[idx],
                    )
                )
            f.write(
                "\n{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".format(
                    test_loss, precision, recall, f1
                )
            )

        agent.update_indices(model)
        import pdb; pdb.set_trace()
        agent.update_datasets(model)