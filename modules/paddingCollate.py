# import torch
# from torch.utils.data._utils.collate import default_collate

# class PaddingCollate:
#     def __init__(self, pad_keys, pad_value=0.0):
#         """
#         Initialize the PaddingCollate instance with specified keys to pad and a padding value.
        
#         Args:
#         - pad_keys (list of str): List of keys that need padding.
#         - pad_value (float): The value used for padding numeric data. Default is 0.0.
#         """
#         self.pad_keys = pad_keys
#         self.pad_value = pad_value

#     def pad_tensor(self, tensor, max_len):
#         """
#         Pad the tensor to the maximum length in the batch.

#         Args:
#         - tensor (torch.Tensor): The tensor to pad.
#         - max_len (int): The maximum length to pad to.

#         Returns:
#         - torch.Tensor: The padded tensor.
#         """
#         # if tensor.size(0) < max_len:
#         #     padding_size = (max_len - tensor.size(0),) + tuple(tensor.shape[1:])
#         #     padding = torch.full(padding_size, self.pad_value, dtype=tensor.dtype, device=tensor.device)
#         #     tensor = torch.cat([tensor, padding], dim=0)
#         if tensor.dim() == 1:  # Simple 1D padding
#             if tensor.size(0) < max_len:
#                 padding_size = (max_len - tensor.size(0),)
#                 padding = torch.full(padding_size, self.pad_value, dtype=tensor.dtype, device=tensor.device)
#                 tensor = torch.cat([tensor, padding], dim=0)
#         else:  # Multidimensional padding
#             padding_size = [0] * (2 * len(tensor.shape))
#             padding_size[-1] = max_len - tensor.size(0)  # Pad the first dimension
#             tensor = torch.nn.functional.pad(tensor, padding_size, value=self.pad_value)
#         return tensor

#     def collate(self, batch):
#         """
#         Collate the batch, padding specified tensors to the same size.

#         Args:
#         - batch (list): A list of dictionaries containing the data samples.

#         Returns:
#         - dict: A dictionary containing the collated data with all specified tensors padded to the same size.
#         """
#         # Find the maximum length for each tensor key that needs padding
#         max_lens = {key: max(x[key].size(0) for x in batch if key in x) for key in self.pad_keys}

#         # Pad each tensor that needs padding
#         for data in batch:
#             for key in self.pad_keys:
#                 if key in data and data[key].size(0) < max_lens[key]:
#                     data[key] = self.pad_tensor(data[key], max_lens[key])
#         # # Ensure all tensors for each key have the same shape
#         # uniform_shapes = {}
#         # for key in self.pad_keys:
#         #     shapes = {data[key].shape for data in batch if key in data}
#         #     if len(shapes) > 1:
#         #         max_shape = tuple(max(sizes) for sizes in zip(*shapes))
#         #         for data in batch:
#         #             if key in data:
#         #                 data[key] = self.pad_tensor(data[key], max_shape[0])           

#         return default_collate(batch)

#     def __call__(self, batch):
#         return self.collate(batch)
# modules/paddingCollate.py
import torch
from torch.nn.utils.rnn import pad_sequence

class PaddingCollate:
    def __init__(self, pad_keys=None, pad_id: int = 0, dist_ignore_index: int = -100):
        # pad_keys 仅为兼容旧调用，不再使用
        self.pad_id = pad_id
        self.dist_ignore_index = dist_ignore_index

    def __call__(self, batch):
        # 文本/ID字段：保持 list，避免进入默认 collate
        ids = [b["id"] for b in batch]
        peptide_seq_text = [b["peptide_seq"] for b in batch]
        receptor_seq_text = [b["receptor_seq"] for b in batch]

        # 1) 序列 token：pad 到同长
        pep_tok = [b["peptide_seq_tensor"].long().contiguous() for b in batch]
        rec_tok = [b["receptor_seq_tensor"].long().contiguous() for b in batch]
        peptide_seq_tensor = pad_sequence(pep_tok, batch_first=True, padding_value=self.pad_id)
        receptor_seq_tensor = pad_sequence(rec_tok, batch_first=True, padding_value=self.pad_id)

        # 2) binding_pocket：每个样本形状 (M_i, 3) → pad 到 (max_M, 3)
        max_M = max(b["binding_pocket"].shape[0] for b in batch) if batch else 0
        bp_list = []
        for b in batch:
            t = b["binding_pocket"].to(torch.float).contiguous()
            m = t.shape[0]
            if m < max_M:
                pad = torch.zeros((max_M - m, 3), dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad], dim=0)
            bp_list.append(t)
        binding_pocket = torch.stack([x.contiguous() for x in bp_list], dim=0)

        # 3) 标量分值 [B, 1]
        vina_affinity = torch.stack([b["vina_affinity"].to(torch.float).contiguous() for b in batch], dim=0)
        stability      = torch.stack([b["stability"].to(torch.float).contiguous()      for b in batch], dim=0)
        solubility     = torch.stack([b["solubility"].to(torch.float).contiguous()     for b in batch], dim=0)

        # 4) 结构标签：每个样本 (L_i, L_i) → pad 到 (max_L, max_L)，用 ignore_index 填充
        max_L = max(b["structure_labels"].shape[0] for b in batch) if batch else 0
        lab_list = []
        for b in batch:
            lab = b["structure_labels"].long().contiguous()
            L = lab.shape[0]
            if L < max_L:
                pad_rows = torch.full((max_L - L, L), self.dist_ignore_index, dtype=lab.dtype, device=lab.device)
                lab = torch.cat([lab, pad_rows], dim=0)  # (max_L, L)
                pad_cols = torch.full((max_L, max_L - L), self.dist_ignore_index, dtype=lab.dtype, device=lab.device)
                lab = torch.cat([lab, pad_cols], dim=1)  # (max_L, max_L)
            lab_list.append(lab)
        structure_labels = torch.stack([x.contiguous() for x in lab_list], dim=0)
        structure_labels_mask = (structure_labels != self.dist_ignore_index)

        return {
            "id": ids,
            "peptide_seq": peptide_seq_text,
            "receptor_seq": receptor_seq_text,
            "peptide_seq_tensor": peptide_seq_tensor,
            "receptor_seq_tensor": receptor_seq_tensor,
            "binding_pocket": binding_pocket,
            "vina_affinity": vina_affinity,
            "stability": stability,
            "solubility": solubility,
            "structure_labels": structure_labels,
            "structure_labels_mask": structure_labels_mask,
        }
