from dataset import Dataset
from pathlib import Path
import logging

if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    dataset = Dataset(
        path_to_dataset='/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/data/train.csv',
        output_dir='/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/data/prepared',
        val_size=0.2,
        config_name='222'
    )
    dataset.apply_training_pipeline()

    test_set = Dataset(
        path_to_dataset='/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/data/test.csv',
        output_dir='/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/data/prepared/test',
        config_name='222'
    )
    test_set.apply_test_pipeline()
