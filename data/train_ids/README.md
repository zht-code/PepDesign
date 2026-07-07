# Training set IDs

This directory contains lightweight training-set ID/list files used by the PepDesign data-preparation workflow.

- `train_nr50_KEEP.txt`: retained training complex IDs after NR50-style filtering.
- `train_nr50_DROP.txt`: removed/excluded IDs during NR50-style filtering.
- `train_nr50_cluster.tsv`: clustering assignment table used for redundancy filtering.
- `train_nr50_rep_seq.fasta`: representative sequences after clustering.
- `train_nr50_all.fasta`: FASTA records for the training set before/around filtering.
- `train_nr50_all_seqs.fasta`: sequence-only FASTA records for the training set.

Large processed tensors, raw structures, and model checkpoints are intentionally excluded from GitHub.
