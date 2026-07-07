import torch

# 生成的奖励函数 (基于你已有的结合打分或预测模型)
def reward_function(predicted_peptide_seq, receptor_embedding):
    # 使用预测模型评估生成的多肽序列的结合亲和性
    # 你可以在这里调用一个打分函数，比如基于虚拟筛选或实验数据
    predicted_binding_score = compute_binding_affinity(predicted_peptide_seq, receptor_embedding)
    return predicted_binding_score



