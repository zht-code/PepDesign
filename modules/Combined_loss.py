import torch.nn as nn
import torch
# 缩放参数
AFFINITY_MIN = -17.4
AFFINITY_MAX = -0.0

# 定义损失函数
class CombinedLoss(nn.Module):
    def __init__(self, binding_loss_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.autoregressiveLoss = AutoregressiveLoss()
        self.affinityLoss = AffinityLoss()
        self.alpha = 0.6  # 亲和力权重
        self.binding_loss_weight = binding_loss_weight
    
    def forward(self, predicted_seq, true_seq):
        # 1. 序列生成的交叉熵损失
        seq_loss = self.autoregressiveLoss(predicted_seq, true_seq)

        # 2. 亲和力损失
        # affinity_loss = self.affinityLoss(predicted_vina_score, true_vina_score)
        
        # 3. 结合现有的亲和力


        # 4. 计算总损失，使用 alpha、beta、gamma 来加权
        # total_loss = seq_loss + self.alpha * affinity_loss

        # return total_loss
        return seq_loss

    # def reinforce_loss(self, log_probs, rewards):
    #     """
    #     REINFORCE策略梯度损失，用于强化学习
    #     :param log_probs: 生成序列时每个时间步的log-probabilities
    #     :param rewards: 奖励信号 (亲和力、稳定性、溶解性等的组合)
    #     """
    #     # 将奖励转换为Tensor，并在同一设备上
    #     rewards = torch.tensor(rewards).to(log_probs[0].device)
        
    #     # 策略梯度损失 (REINFORCE) 是 -log_probs * reward 的总和
    #     loss = -torch.sum(torch.stack(log_probs) * rewards)

    #     return loss

# 定义自回归损失
class AutoregressiveLoss(nn.Module):
    def __init__(self):
        super(AutoregressiveLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predicted_logits, true_seq):
        """
        计算自回归损失
        :param predicted_logits: 模型预测的 logits，形状为 [batch_size, seq_len, vocab_size]
        :param true_seq: 真实的目标序列，形状为 [batch_size, seq_len]
        """
        # 将 predicted_logits 和 true_seq 变形为适合计算损失的形状
        # 预测值变形为 [batch_size * seq_len, vocab_size]
                # 检查并调整predicted_logits以匹配true_seq的长度
        # true_seq_len = true_seq.size(1)
        # attr_seq_len = predicted_logits.size(1)
        # seq_len = attr_seq_len - true_seq_len
        # predicted_logits = predicted_logits[:, 1:, :]  # 只取真实seq_len的预测结果
        predicted_logits = predicted_logits.reshape(-1, predicted_logits.size(-1))
        
        # 真实值变形为 [batch_size * seq_len]
        true_seq = true_seq.view(-1)
        
        # 计算交叉熵损失
        loss = self.ce_loss(predicted_logits, true_seq)
        
        return loss
class AffinityLoss(nn.Module):
    def __init__(self):
        super(AffinityLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_affinity, true_affinity):
        """
        计算亲和力的损失
        :param predicted_affinity: 预测的亲和力值，形状为 [batch_size] 或 [batch_size, 1]
        :param true_affinity: 真实的亲和力值，形状应与 predicted_affinity 相同
        :return: 亲和力损失的标量值
        """
        # 缩放 ΔG 值
        true_vina_score_scaled = scale_affinity(true_affinity)
        # 确保输入的形状正确
        if predicted_affinity.dim() > 1:
            predicted_affinity = predicted_affinity.squeeze(-1)
        if true_vina_score_scaled.dim() > 1:
            true_vina_score_scaled = true_vina_score_scaled[:, 0]

        # 计算均方误差损失
        loss = self.mse_loss(predicted_affinity, true_vina_score_scaled)
        
        return loss

def scale_affinity(delta_g_tensor, min_val=AFFINITY_MIN, max_val=AFFINITY_MAX):
    """
    将ΔG值缩放到0-1范围
    """
    scaled = 1.0 - (delta_g_tensor - min_val) / (max_val - min_val)
    scaled = torch.clamp(scaled, 0.0, 1.0)  # 确保在0-1范围内
    return scaled

def inverse_scale_affinity(scaled_tensor, min_val=AFFINITY_MIN, max_val=AFFINITY_MAX):
    """
    将缩放后的ΔG值逆缩放回原始范围
    """
    delta_g = min_val + (1.0 - scaled_tensor) * (max_val - min_val)
    return delta_g