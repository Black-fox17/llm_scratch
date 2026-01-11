import numpy as np


class Position:
    def __init__(self):
        pass

    def sinusodal_positional_encoding(self, seq_len, d_model):
        pe = np.zeros((seq_len, d_model))
        pos = np.arange(0, seq_len).reshape((seq_len, 1))
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        return pe
        
