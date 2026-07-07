nohup python -u /root/autodl-tmp/Peptide_3D/utils/dpo/train_DPO_multi_objective.py \
  --aff_jsonl /root/autodl-tmp/Peptide_3D/utils/dpo/affinity_pairs_cleaned.jsonl \
  --stab_jsonl /root/autodl-tmp/Peptide_3D/utils/dpo/stability_pairs.jsonl \
  --sol_jsonl  /root/autodl-tmp/Peptide_3D/utils/dpo/solubility_pairs.jsonl \
  --lambda_aff 1.0 --lambda_stab 0.35 --lambda_sol 0.35 \
  --normalize_lambda \
  --init_ckpt /root/autodl-tmp/Peptide_3D/logs_Ranger_no_DPO/best_model_epoch_72_loss_2.0048.pth \
  --save_dir /root/autodl-tmp/Peptide_3D/logs_Ranger_dpo_multi \
  --epochs 5 --batch_size 1 --grad_accum 4 \
  --dpo_mode soft --kl_coef 0.01 --kl_pairs 2048 \
  --use_amp --save_every_epoch \
  > /root/autodl-tmp/Peptide_3D/utils/dpo/dpo_multi.log 2>&1 &
