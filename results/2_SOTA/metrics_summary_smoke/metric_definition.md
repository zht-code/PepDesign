# 2_SOTA Unified Metric Definitions

- `hdock_score`: parsed from existing HDOCK result files / json payloads. Lower is better.
- `contact_consistency`: Jaccard overlap of receptor-peptide interface residue contacts between native complex and predicted top1 docked complex, reusing project `contact_map_consistency()`.
- `plddt`: mean B-factor value read directly from the peptide chain atoms of the candidate structure.
- `ramachandran_compliance`: fraction of peptide residues whose phi/psi angles fall into project-favored Ramachandran regions.
- `clash_score`: project heavy-atom clash score, reported as clashes per 1000 atoms with 0.4 A VDW overlap tolerance; invalid structures return `NaN`.
- `perplexity`: placeholder column. The current repository does not expose a ready-to-call peptide LM scorer, so values remain `NaN` unless a scorer is implemented later.
- `repetition_rate`: repeated 3-gram ratio over the peptide sequence, i.e. fraction of overlapping 3-mers belonging to a 3-mer type observed more than once.
- `train_similarity`: nearest-neighbor sequence identity against the training set, computed with MMseqs2 (`easy-search`, best hit `pident` / 100).
- `novelty`: binary flag derived from `train_similarity < novelty_threshold`.
- `novelty_ratio`: mean of `novelty` during aggregation.

## Implementation Notes

- Native complexes are cached under `_native_complex_cache` by merging receptor and native peptide PDBs.
- For multi-chain candidate PDBs, peptide-only structure metrics and sequence extraction are computed on the shortest chain, or the chain whose length is closest to the native peptide length when available.
- Existing project modules were reused where available: affinity contact consistency, PDB loading, pLDDT reading, Ramachandran compliance, clash score, and native complex merging.
