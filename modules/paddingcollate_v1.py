import torch
import torch
from torch.utils.data._utils.collate import default_collate

class PaddingCollate:
    def __init__(self, pad_keys, pad_value=0.0, pad_string="<pad>", num_gpus=3):
        """
        Initialize the PaddingCollate instance with specified keys to pad, a padding value for tensors,
        a padding string for lists, and the number of GPUs.

        Args:
        - pad_keys (list of str): List of keys that need padding.
        - pad_value (float): The value used for padding numeric data.
        - pad_string (str): The string used for padding string lists.
        - num_gpus (int): Number of GPUs available for distributing the data.
        """
        self.pad_keys = pad_keys
        self.pad_value = pad_value
        self.pad_string = pad_string
        self.num_gpus = num_gpus

    def pad_tensor(self, tensor, max_len):
        if tensor.dim() == 1:
            padding_size = max_len - tensor.size(0)
            if padding_size > 0:
                tensor = torch.cat([tensor, torch.full((padding_size,), self.pad_value, device=tensor.device)])
        return tensor

    def pad_string_list(self, data_list, max_len):
        return data_list + [self.pad_string] * (max_len - len(data_list))

    def collate(self, batch):
        # Calculate max lengths for padding
        max_lens = {}
        for key in self.pad_keys:
            max_len = 0
            for item in batch:
                if key in item:
                    data = item[key]
                    if isinstance(data, list):
                        max_len = max(max_len, len(data))
                    elif isinstance(data, torch.Tensor):
                        max_len = max(max_len, data.size(0))
            max_lens[key] = max_len

        # Pad and split data for each key
        for item in batch:
            for key in self.pad_keys:
                if key in item:
                    data = item[key]
                    if isinstance(data, list):
                        item[key] = self.pad_string_list(data, max_lens[key])
                    elif isinstance(data, torch.Tensor):
                        item[key] = self.pad_tensor(data, max_lens[key])

        # If multiple GPUs are available, split data across GPUs
        if self.num_gpus > 1:
            new_batch = []
            for gpu in range(self.num_gpus):
                part = [{} for _ in range(len(batch) // self.num_gpus + (1 if gpu < len(batch) % self.num_gpus else 0))]
                for i, item in enumerate(part):
                    idx = gpu * (len(batch) // self.num_gpus) + i
                    for key in batch[idx]:
                        item[key] = batch[idx][key][gpu::self.num_gpus] if isinstance(batch[idx][key], torch.Tensor) else batch[idx][key]
                new_batch.extend(part)
            batch = new_batch

        return default_collate(batch)

    def __call__(self, batch):
        return self.collate(batch)

