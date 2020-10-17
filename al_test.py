from train import *
from active_learning import ActiveLearningDataset

if __name__ == '__main__':

    a = ActiveLearningDataset(train_data, test_data, 5, 6000, args.batch_size)
    a.update_indices(model)