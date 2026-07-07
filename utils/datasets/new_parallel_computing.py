
from vina_score_computing import *
import multiprocessing
import os
import traceback
import json

def process_subfolder(subFolder, path_to_dataSet_folder, parser):
    path_to_single_folder = os.path.join(path_to_dataSet_folder, subFolder)
    
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
            
            return subFolder, score_list  # 返回文件夹名称和得分列表

        except Exception as e:
            error_message = f"Error processing {subFolder}: {str(e)}\n{traceback.format_exc()}"
            return subFolder, error_message

    return subFolder, "Not a directory"

def worker(task_queue, results, parser):
    while True:
        subFolder = task_queue.get()
        if subFolder is None:  # 结束信号
            break
        result = process_subfolder(subFolder, dataset_path, parser)
        results.append(result)  # 将结果添加到共享列表

def load_existing_results(output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return json.load(f)
    return []

def parallel_process_and_save(folder_path, output_file='results.json', num_processes=260):
    manager = multiprocessing.Manager()
    results = manager.list()  # 使用 Manager 创建一个共享列表
    task_queue = multiprocessing.Queue()

    parser = PDB.PDBParser(QUIET=True)

    # 读取已处理的文件夹
    existing_results = load_existing_results(output_file)
    processed_folders = {result['folder'] for result in existing_results}  # 假设结果字典中有 'folder' 作为键

    # 将所有子文件夹放入任务队列，排除已处理的文件夹
    for dataSetFolder in os.listdir(folder_path):
        path_to_dataSet_folder = os.path.join(folder_path, dataSetFolder)
        subfolders = [f for f in os.listdir(path_to_dataSet_folder) if os.path.isdir(os.path.join(path_to_dataSet_folder, f))]
        for subfolder in subfolders:
            if subfolder not in processed_folders:  # 只添加未处理的文件夹
                task_queue.put(subfolder)

    # 启动工作进程
    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(task_queue, results, parser))
        p.start()
        processes.append(p)

    # 发送结束信号
    for _ in range(num_processes):
        task_queue.put(None)

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 将结果写入 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(list(results), f, indent=2)

    print("All data processed and saved to", output_file)

if __name__ == "__main__":
    dataset_path = '/root/autodl-tmp/PPDbench'
    parallel_process_and_save(dataset_path, output_file='results.json', num_processes=260)