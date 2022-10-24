from models import BidirectionalLSTM
from pathlib import Path
import pickle

if __name__ == '__main__':
    input_dir = Path('/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/data/prepared')
    output_dir = Path('/Users/julienclement/Documents/Projects/kaggle-novozymes-enzyme-stability-prediction/ml/output')

    with open(input_dir / 'X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)

    with open(input_dir / 'y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    with open(input_dir / 'X_val.pkl', 'rb') as f:
        X_val = pickle.load(f)

    with open(input_dir / 'y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)

    model = BidirectionalLSTM(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config_name='default'
    )

    model.fit(output_dir=output_dir)
    model.export(output_dir=output_dir)
    model.score_self()
    model.plot(output_dir=output_dir)
