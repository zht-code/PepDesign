"""
author: felx
date: 2024.10.04
desc: compute the vina score
"""
from Bio import PDB
import numpy as np
import os
import subprocess
import re
import json
import traceback

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
            if 'CA' not in residue:
                continue  # 跳过没有 CA 原子的残基
            # 计算坐标的距离 
            distance = residue['CA'].coord - peptide_coord
            if np.linalg.norm(distance) <= threshold:
                # 如果在阈值内，保存受体残基坐标(CA为中心)
                # 把坐标转变为tuple便于后续对接
                interaction_windows.append(tuple(residue['CA'].coord))
                break  # 找到一个相互作用后，可以跳出循环
    return interaction_windows

def pdb_to_pdbqt(pdb_file, pdbqt_file, is_receptor=True):
    """将 PDB 文件转换为 PDBQT 文件"""
    script = 'prepare_receptor4.py' if is_receptor else 'prepare_ligand4.py'
    command = [
        '/root/mgltools_x86_64Linux2_1.5.7/bin/pythonsh',
        f'/root/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/{script}',
        '-r' if is_receptor else '-l', pdb_file,
        '-o', pdbqt_file
    ]
    print(f"Running command: {' '.join(command)}")

    # 运行命令并捕获输出和错误信息
    result = subprocess.run(command, capture_output=True, text=True)
    
    # # 打印详细的输出和错误信息
    # print(f"Command output: {result.stdout}")
    print(f"Command error: {result.stderr}")
    
    # if result.returncode != 0:
    #     raise RuntimeError(f"Error converting {pdb_file} to {pdbqt_file}: {result.stderr}")
    
    print(f'Wrote converted file to {pdbqt_file}')
    return pdbqt_file

def run_qvina_and_parse(path_to_single_folder, receptor_pdbqt, ligand_pdbqt, center_x, center_y, center_z , score_list):
    """运行 QVina 并解析日志文件，获取最好的 affinity 最后删除临时文件"""
    #exhaustiveness 8搜索广度 已知结合位点
    command = [
        '/root/qvina/qvina-master/bin/qvina02',
        '--receptor', receptor_pdbqt,
        '--ligand', ligand_pdbqt,
        f'--center_x', str(center_x),
        f'--center_y', str(center_y),
        f'--center_z', str(center_z),
        f'--size_x', '25',
        f'--size_y', '25',
        f'--size_z', '25',
        '--exhaustiveness', '8'
    ]
    
    print(f"Running command: {' '.join(command)}")
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    # print(f"Command output: {result.stdout}")
    print(f"Command error: {result.stderr}")
    
    # if result.returncode != 0:
    #     raise RuntimeError(f"Error running qvina: {result.stderr}")
    
    print(f'QVina run complete.')
    
     # 解析 result.output，获取最好的 affinity
    best_affinity = None
    affinity_pattern = r"^\s*\d+\s+(-\d+\.\d+)"  # 匹配 affinity 的正则表达式
    matches = re.findall(affinity_pattern, result.stdout, re.MULTILINE)
    
    if matches:
        affinities = [float(match) for match in matches]
        best_affinity = min(affinities)  # affinity 越负越好，所以选择最小值
        print(f"Best affinity: {best_affinity}")
    else:
        print("No affinities found in the output")
    # 直接保存 到后面去选最小的score
    score_list.append({
        "folder": path_to_single_folder,
        "center": (center_x,center_y,center_z),
        "score": best_affinity
    })

# 加载已存在的结果
def load_existing_results(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

# 保存数据到文件
def save_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to {filename}")

# 追加结果到文件
def append_scores_to_data(score_list, big_dict):
    for entry in score_list:
        id_value = entry['folder'].split('/')[-1]  # 获取文件夹名称作为 id
        center = {
            "center_x": float(entry['center'][0]),
            "center_y": float(entry['center'][1]),
            "center_z": float(entry['center'][2])
        }
        score = entry['score']
        if score is None:
            continue
        
        if id_value in big_dict:
            existing_scores = big_dict[id_value]
            min_existing_score = min(existing_scores, key=lambda x: x['score'])['score']

            if score < min_existing_score:
                big_dict[id_value] = [{"center": center, "score": score}]
        else:
            big_dict[id_value] = [{"center": center, "score": score}]
    
    return big_dict

def run_docking_process(path_to_single_folder, receptor_pdb, peptide_pdb, interaction_windows, score_list):
    """完成 PDB 文件转换、运行 QVina、解析日志并删除临时文件, 最后追加结果到文件夹"""
    
    # 将 receptor 和 ligand 转换为 PDBQT
    receptor_pdbqt = pdb_to_pdbqt(receptor_pdb, os.path.join(path_to_single_folder, 'receptor.pdbqt'), is_receptor=True)
    ligand_pdbqt = pdb_to_pdbqt(peptide_pdb, os.path.join(path_to_single_folder, 'peptide.pdbqt'), is_receptor=False)
    
    # 针对每个结合窗口运行QVina windows中应该是个三元组(x,y,z)
    for window in interaction_windows:
        center_x, center_y, center_z = window
        
        # 运行 qvina 并立即解析 log 文件
        run_qvina_and_parse(
            path_to_single_folder,
            receptor_pdbqt,
            ligand_pdbqt,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            score_list=score_list
        )
    
    # 删除 PDBQT 文件
    if os.path.exists(receptor_pdbqt):
        os.remove(receptor_pdbqt)
        print(f"Deleted receptor PDBQT file: {receptor_pdbqt}")
    if os.path.exists(ligand_pdbqt):
        os.remove(ligand_pdbqt)
        print(f"Deleted ligand PDBQT file: {ligand_pdbqt}")
    #删除out输出文件
    out_pdbqt = os.path.join(path_to_single_folder, 'peptide_out.pdbqt')
    if os.path.exists(out_pdbqt):
        os.remove(out_pdbqt)
        print(f"Deleted out PDBQT file: {out_pdbqt}")

# 主要处理和保存函数
def process_and_save(folder_path, batch_size=10, output_file='result.json'):
    processed_count = 0
    batch_number = 1
    results = load_existing_results(output_file)
    skipped_folders = []

    parser = PDB.PDBParser(QUIET=True)

    for dataSetFolder in os.listdir(folder_path):
        path_to_dataSet_folder = os.path.join(folder_path, dataSetFolder)
        for subFolder in os.listdir(path_to_dataSet_folder):
            if subFolder in ["ligands133", "receptors133", "results"]:
                continue
            
            path_to_single_folder = os.path.join(path_to_dataSet_folder, subFolder)
            
            # 跳过已处理的文件夹
            if subFolder in results:
                skipped_folders.append(f"Skipped {subFolder}: Already processed")
                continue

            if os.path.isdir(path_to_single_folder):
                try:
                    receptor_pdb = os.path.join(path_to_single_folder, "receptor.pdb")
                    peptide_pdb = os.path.join(path_to_single_folder, "peptide.pdb")
                    
                    # 读取蛋白质结构
                    peptide_structure = parser.get_structure("peptide", peptide_pdb)
                    receptor_structure = parser.get_structure("receptor", receptor_pdb)
                    
                    # 过滤掉水分子和离子
                    filtered_peptide_structure = filter_structure_without_h2oNion(peptide_structure, "peptide")
                    filtered_receptor_structure = filter_structure_without_h2oNion(receptor_structure, "receptor")
                    
                    # 获取相互作用窗口
                    threshold = 4.0
                    interaction_windows = get_interaction_windows(threshold, filtered_receptor_structure, filtered_peptide_structure)

                    score_list = []
                    # 运行对接过程
                    run_docking_process(path_to_single_folder, receptor_pdb, peptide_pdb, interaction_windows, score_list)
                    
                    # 更新结果
                    results = append_scores_to_data(score_list, results)

                    processed_count += 1
                    print(f"Processed {subFolder}")

                    # 批量保存
                    if processed_count % batch_size == 0:
                        save_to_file(results, output_file)
                        print(f"Completed and saved batch {batch_number}")
                        batch_number += 1
                        results = load_existing_results(output_file)

                except Exception as e:
                    error_message = f"Error processing {subFolder}: {str(e)}\n{traceback.format_exc()}"
                    print(error_message)
                    skipped_folders.append(error_message)

    # 保存最后一批数据（如果有的话）
    if processed_count % batch_size != 0:
        save_to_file(results, output_file)
        print(f"Saved final batch")

    print("All data processed and saved.")
    print("\nSkipped folders:")
    for folder in skipped_folders:
        print(folder)

if __name__ == "__main__":
    dataset_path = '../dataset/'
    process_and_save(dataset_path, batch_size=10)