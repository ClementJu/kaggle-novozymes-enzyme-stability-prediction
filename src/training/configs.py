configs = {
    'bi_lstm': {
        'default': {
            'n_epochs': 120,
            'batch_size': 256,
            'sequence_length': 222,
            'l2_reg': 0.01,
            'dropout': 0.3,
            'learning_rate': 1e-4
        }
    }
}


class BiLstmConfig:
    def __init__(self, n_epochs: float, batch_size: int,
                 sequence_length: int, l2_reg: float,
                 dropout: float, learning_rate: float):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.learning_rate = learning_rate

    @classmethod
    def get_config(cls, config_name):
        config = configs['bi_lstm'][config_name]
        return cls(**config)

    def as_fit_config_dict(self):
        return {
            'epochs': self.n_epochs,
            'batch_size': self.batch_size
        }

