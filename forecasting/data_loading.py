import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class UnivariateTimeSeries(Dataset):
    def __init__(self, ts, sequence_length, prediction_length):
        self.ts = ts.astype(np.float32)
        self.sequence_length = sequence_length
        sequences = np.row_stack([ts[i * self.sequence_length:(i + 1) * self.sequence_length] for i in
                                  range(0, int((len(self.ts)) / self.sequence_length))])
        prediction_sequences = np.apply_along_axis(lambda x: x[-prediction_length:], axis=1, arr=sequences)
        self.sequences = torch.from_numpy(np.nan_to_num(sequences))
        self.prediction_sequences = torch.from_numpy(np.nan_to_num(prediction_sequences))

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, j):
        return self.sequences[j, :], self.prediction_sequences[j, :]


def get_dataloader(ts, sequence_length, prediction_length, bach_size, shuffle=False):
    ts_obj = UnivariateTimeSeries(ts, sequence_length, prediction_length)
    return DataLoader(ts_obj, batch_size=bach_size, shuffle=shuffle)
