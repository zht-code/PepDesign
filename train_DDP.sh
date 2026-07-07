cd /root/autodl-tmp/Peptide_3D

# 可选：限制用哪几张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 可选：单机时禁用 IB，避免 NCCL 报错
export NCCL_IB_DISABLE=1
# 可选：减少 CPU 线程争用
export OMP_NUM_THREADS=4

# 跑起来（把 stdout/stderr 写到带时间戳的日志）
nohup torchrun --nproc_per_node=4 --master_port=29501 \
  /root/autodl-tmp/Peptide_3D/train_DDP.py \
  > /root/autodl-tmp/Peptide_3D/train_$(date +%F_%H-%M-%S).log 2>&1 &

