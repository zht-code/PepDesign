import os
from torch.utils.data import DataLoader

# 确保正确加载模块路径
from modules.Dataset import ProteinPeptideDataset

class DatasetProcess:
    def __init__(self, pdb_dir, json_files, batch_size=4, collate_fn=None, shuffle=True, num_workers=2):
        """
        初始化 DatasetProcess 类
        :param pdb_dir: PDB 文件所在的目录
        :param batch_size: DataLoader 的批量大小
        :param shuffle: 是否在每个 epoch 后 shuffle 数据
        :param num_workers: DataLoader 的并行线程数
        """
        self.pdb_dir = pdb_dir  # PDB 文件所在路径
        self.json_files = json_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.id_list = self.get_pdb_id()  # 获取所有的 ID 列表
        self.data_loader = None  # 初始化 DataLoader 为空
        self.collate_fn = collate_fn

    def get_pdb_id(self):
        """
        获取指定目录中所有文件夹的名字作为 PDB ID
        :return: PDB ID 列表
        """
        # 获取 PDB 文件夹中的所有文件夹名称
        pdb_id = [name for name in os.listdir(self.pdb_dir) if os.path.isdir(os.path.join(self.pdb_dir, name))]
        return pdb_id

    def create_dataset(self):
        """
        创建 ProteinPeptideDataset 实例
        :return: dataset 对象
        """
        # 根据获取到的 ID 列表创建 ProteinPeptideDataset 数据集实例
        dataset = ProteinPeptideDataset(self.id_list, self.pdb_dir, self.json_files, interface_json='/root/autodl-tmp/Peptide_3D/data/interface_labels.json')
        return dataset

    def create_data_loader(self):
        """
        创建 DataLoader 实例
        :return: DataLoader 对象
        """
        dataset = self.create_dataset()  # 创建数据集
        # 根据数据集创建 DataLoader 实例
        # self.data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=self.shuffle, num_workers=self.num_workers)
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=False  # <—— 加上这行
        )

        return self.data_loader
