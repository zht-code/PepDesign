import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn

# pos_heavyatom: 重原子的坐标列表
def normalize_coordinates(coords):
    coords = np.array(coords)
    mean = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    return (coords - mean) / std

normalized_coords = normalize_coordinates([ 
    [15.491,  7.05 , 17.165],
    [14.946,  7.55 , 15.888],
    [16.011,  7.604, 14.814],
    [17.199,  7.619, 15.115],
])

print(normalized_coords)  # 正则化后的坐标
# 二面角
def encode_dihedral_angle(angle):
    """将角度转换为sin和cos形式，适应周期性处理"""
    return np.sin(angle), np.cos(angle)
# 示例：处理phi角度
phi_angles = np.array([-44.08647423, -50.50514144])
phi_encoded = [encode_dihedral_angle(angle) for angle in phi_angles]
phi_encoded = np.concatenate(phi_encoded)  # 将sin和cos值拼接成一维向量

# 示例氨基酸类型
aa_type = 'GLY'  # 氨基酸三字母代码
aa_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLY']  # 所有可能的氨基酸类型
# One-hot编码
encoder = OneHotEncoder(sparse=False)
aa_encoded = encoder.fit_transform([[aa_type]])
print(aa_encoded)

# 假设已经分别计算出每个特征
chain_id_encoded = [0, 1]  # 示例：链ID经过编码
res_nb = 1  # 示例残基编号
aa_encoded = aa_encoded.flatten()  # 氨基酸类型编码
bfactor_encoded = np.array([bfactor_normalized])  # B因子
dihedral_angles_encoded = np.concatenate([phi_encoded, psi_encoded, omega_encoded])  # 角度编码
coords_encoded = normalized_coords.flatten()  # 原子坐标

# 将所有特征拼接成一个完整的向量
input_vector = np.concatenate([chain_id_encoded, [res_nb], aa_encoded, bfactor_encoded, dihedral_angles_encoded, coords_encoded])


# 假设input_vector是从特征中提取和拼接的输入向量
input_tensor = torch.tensor(input_vector, dtype=torch.float32)

# 简单的全连接层
fc = nn.Linear(input_tensor.size(0), 128)
output = fc(input_tensor)

print(output.shape)  # 输出维度