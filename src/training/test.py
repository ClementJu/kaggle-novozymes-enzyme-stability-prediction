from models import BidirectionalLSTM
from pathlib import Path
import pickle

if __name__ == '__main__':
    path_to_test_set = Path('/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/data/prepared/test/X_test.pkl')
    path_to_ids = Path('/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/data/prepared/test/X_seq_id.pkl')
    path_to_model = Path('/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/ml/output/model.h5')
    path_output_dir = Path('/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/ml/output')

    with open(path_to_test_set, 'rb') as f:
        X_test = pickle.load(f)

    model = BidirectionalLSTM.for_prediction()
    model.apply_prediction_pipeline(path_to_model=path_to_model,
                                    path_to_test_set=path_to_test_set,
                                    path_to_ids=path_to_ids,
                                    path_output_dir=path_output_dir)
