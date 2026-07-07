"""
author: felx
date: 2024.9.23
desc: find the window area between peptide and receptor
"""
from Bio import PDB
import numpy as np
import os

# 自定义选择器 用于过滤
class MySelect(PDB.Select):
    '''预处理蛋白质---选择类'''
    def accept_residue(self, residue):
    # 过滤 保留主要的氨基酸残基
    # 只接受非水分子和非离子
        return residue.get_resname() not in ['HOH', 'Na', 'Cl', 'K', 'Mg', 'Ca']
        
#过滤函数    
def filter_structure_without_h2oNion(protein_structure,protein_name):
    '''预处理蛋白质---过滤水和离子'''
    filtered_protein_structure = PDB.Structure.Structure("filtered_"+protein_name)
    for model in protein_structure:
        new_model = PDB.Model.Model(model.id)
        for chain in model:
            new_chain = PDB.Chain.Chain(chain.id)
            for residue in chain:
                if MySelect().accept_residue(residue):
                    new_chain.add(residue.copy())  # 添加过滤后的残基
            if len(new_chain) > 0:
                new_model.add(new_chain)
        filtered_protein_structure.add(new_model)
    return filtered_protein_structure


def get_interaction_windows(threshold,filtered_receptor_structure,filtered_peptide_structure):
    '''获取窗口位置坐标'''
    #获取多肽原子位置
    peptide_atoms = [atom for model in filtered_peptide_structure for chain in model for residue in chain for atom in residue]
    peptide_coords = [atom.coord for atom in peptide_atoms]

     # 获取受体的残基 is_aa保证是氨基酸
    receptor_residues = [residue for model in filtered_receptor_structure for chain in model for residue in chain if PDB.is_aa(residue)]

    interaction_windows = []
    for residue in receptor_residues:
        for peptide_coord in peptide_coords:
            # 计算坐标的距离 
            distance = residue['CA'].coord - peptide_coord
            if np.linalg.norm(distance) <= threshold:
                # 如果在阈值内，保存受体残基坐标(CA为中心)
                interaction_windows.append(residue['CA'].coord)
                break  # 找到一个相互作用后，可以跳出循环
    return interaction_windows

#程序入口
if __name__ == "__main__":
    parser = PDB.PDBParser(QUIET=True)
    folder_path = './dataset/'
    for dataSetFolder in os.listdir(folder_path):
        path_to_dataSet_folder = os.path.join(folder_path,dataSetFolder)
        for subFolder in os.listdir(path_to_dataSet_folder):
            #保证不动原来目录
            if(subFolder not in ["ligands133","receptors133","results"]):
                path_to_single_folder = os.path.join(path_to_dataSet_folder,subFolder)
                #检查是否是目录 还得过滤是否是原始数据啥的 要不就直接删了
                if os.path.isdir(path_to_single_folder):
                    #不管fix后的，直接用原始的
                    peptide_structure = parser.get_structure("peptide",os.path.join(path_to_single_folder,"peptide.pdb"))
                    receptor_structure = parser.get_structure("reptor",os.path.join(path_to_single_folder,"receptor.pdb"))
                    #过滤水和离子
                    filtered_peptide_structure = filter_structure_without_h2oNion(peptide_structure,"peptide")
                    filtered_receptor_structure = filter_structure_without_h2oNion(receptor_structure,"receptor")
                    #默认距离阈值为5A
                    threshold = 5.0
                    interaction_windows = get_interaction_windows(threshold,filtered_receptor_structure,filtered_peptide_structure)
                    #append到一个文件中 指定结果文件路径 名称用父文件夹 
                    file_path = 'result.txt'
                    if not os.path.exists(file_path):
                        open(file_path, 'a').close()  # 创建空文件
                    with open(file_path, 'a') as f:
                        np.savetxt(f, interaction_windows, fmt='%f', header="file_name:"+subFolder, comments='',delimiter=',')
                        #日志
                        print("dealing "+subFolder+" data done")
                        print('-------------------------------')
            

