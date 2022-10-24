configs = {
    'default': {
        'seq_len_upper_bound_drop': None,
        'max_seq_len': None
    },
    '222': {
        'seq_len_upper_bound_drop': None,
        'max_seq_len': 222
    }
}

class Config:
    def __init__(self, seq_len_upper_bound_drop: int, max_seq_len: int):
        self.seq_len_upper_bound_drop = seq_len_upper_bound_drop
        self.max_seq_len = max_seq_len
