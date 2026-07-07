nohup python /root/autodl-tmp/Peptide_3D/utils/dpo/train_DPO.py \
  --jsonl /root/autodl-tmp/Peptide_3D/utils/dpo/dpo_pairs_cleaned.jsonl \
  --init_ckpt /root/autodl-tmp/Peptide_3D/logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth \
  --save_dir /root/autodl-tmp/Peptide_3D/logs_Ranger_dpo \
  --epochs 5 --batch_size 1 --max_receptor_len 512 --reduce_mode mean \
  --save_every_epoch  \
  > /root/autodl-tmp/Peptide_3D/utils/dpo/train_DPO.log 2>&1 &