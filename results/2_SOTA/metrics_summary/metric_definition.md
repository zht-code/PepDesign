# 2_SOTA Unified Metric Definitions

- `hdock_score`: parsed from existing HDOCK result files / json payloads. Lower is better.
- `contact_consistency`: Jaccard overlap of receptor-peptide interface residue contacts between native complex and predicted top1 docked complex, reusing project `contact_map_consistency()`.
- `plddt`: recomputed with ESMFold from the candidate peptide sequence. The reported value is the mean pLDDT read from the ESMFold output PDB B-factors.
- `ramachandran_compliance`: recomputed with MolProbity Ramachandran analysis (`mmtbx.validation.ramalyze`) on the peptide-only PDB, reported as the favored-residue fraction.
- `clash_score`: recomputed with MolProbity clashscore on the peptide-only PDB after hydrogen addition with Reduce, reported as clashes per 1000 atoms.
- `perplexity`: placeholder column. The current repository does not expose a ready-to-call peptide LM scorer, so values remain `NaN` unless a scorer is implemented later.
- `repetition_rate`: repeated 3-gram ratio over the peptide sequence, i.e. fraction of overlapping 3-mers belonging to a 3-mer type observed more than once.
- `train_similarity`: nearest-neighbor sequence identity against the training set, computed with MMseqs2 (`easy-search`, best hit `pident` / 100). By default the training pool is built from `/autodl-tmp/train_data` peptide files. If exhaustive MMseqs2 search still finds no detectable hit, the similarity is recorded as `0.0`.
- `novelty`: binary flag derived from `train_similarity < novelty_threshold`.
- `novelty_ratio`: mean of `novelty` during aggregation.

## Implementation Notes

- Native complexes are cached under `_native_complex_cache` by merging receptor and native peptide PDBs.
- For multi-chain candidate PDBs, peptide-only structure metrics and sequence extraction are computed on the shortest chain, or the chain whose length is closest to the native peptide length when available.
- Existing project modules were reused where available for indexing, sequence extraction, peptide-chain selection, affinity contact consistency, and native complex merging.
