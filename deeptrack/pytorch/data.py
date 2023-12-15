import torch
import torch.nn as nn

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, pipline, inputs=None, length=None, replace=False):
        # self.dataset = dataset


        if inputs is None:
            if length is None:
                raise ValueError("Either inputs or length must be specified.")
            else:
                self._inp
                inputs = [{}] * length

        self.data = [None for _ in range(self.length)]

    def __getitem__(self, index):
        if self.data[index] is None:
            res =  self.dataset.resolve(**self.inputs[index])
            if not isinstance(res, (tuple, list)):
                res = (res, )
            res = (torch.tensor(r) for r in res)

            self.data[index] = res

        return self.data[index]

    def __len__(self):
        return len(self.dataset)