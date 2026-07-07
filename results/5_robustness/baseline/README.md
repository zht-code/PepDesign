# Baseline Robustness Workspace

This directory contains the full landing area for baseline robustness evaluation under the same target perturbation framework as `results/5_robustness`.

## Official Repositories
- `RFdiffusion`: https://github.com/RosettaCommons/RFdiffusion.git, commit `9535f1938203a24937d7dadf0cb831d02cb5fc0e`, cloned `2026-04-15T04:48:37.011205+00:00`, purpose: Official RFdiffusion codebase used for provenance and optional post-processing.
- `protein_generator`: https://github.com/RosettaCommons/protein_generator.git, commit `94b13b02f73f1cfe774ee72f69bbe5991363550d`, cloned `2026-04-15T04:49:27.775501+00:00`, purpose: Official ProteinGenerator repository for provenance and adapter reference.
- `BindCraft`: https://github.com/martinpacesa/BindCraft.git, commit `20f47e71308a7893e22be3063ae66c76d303cb19`, cloned `2026-04-15T04:49:32.491529+00:00`, purpose: Official BindCraft repository for provenance and format reference.
- `ProteinMPNN`: https://github.com/dauparas/ProteinMPNN.git, commit `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`, cloned `2026-04-15T04:54:05.636918+00:00`, purpose: Official ProteinMPNN repository for RFdiffusion sequence recovery when needed.

## Existing Baseline Result Sources
- `bindcraft`: rows=665, resolved=665, comparable_target_sets=['ppdbench'], sample_source_dirs=['/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/cache/clean_inputs/bindcraft/1cjr']
- `proteingenerator`: rows=665, resolved=402, comparable_target_sets=['ppdbench'], sample_source_dirs=['/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/cache/clean_inputs/proteingenerator/1cjr']
- `rfdiffusion`: rows=665, resolved=665, comparable_target_sets=['ppdbench'], sample_source_dirs=['/root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/cache/clean_inputs/rfdiffusion/1cjr']

## Coverage Notes
- `ProteinGenerator`: direct PPDbench peptide candidates were found and can be re-evaluated in the shared robustness pipeline.
- `BindCraft`: the machine contains family-level outputs and JSON references to PPDbench outputs, but the referenced PPDbench peptide PDB files were not present at the expected paths during indexing. These candidates are therefore logged as unavailable for the strict PPDbench comparison unless matching files are later restored.
- `RFdiffusion`: no direct PPDbench peptide candidate set was found on the current machine; family/protein split artifacts are intentionally excluded from the main comparison because they do not match the robustness target IDs used by `ours`.

## Unified Evaluation Definition
- Perturbations are applied to the target protein, not to the already generated peptide candidates.
- Perturbation families are aligned to the existing Chapter 5 setup: `structure_missing` = 0/10/20/30/40%, `pocket_noise` = 0/0.5/1.0/1.5/2.0 A, `sequence_trunc` = 0/10/20/30/40%.
- Metrics reuse the same project-side definitions whenever possible: `affinity_hdock`, `stability`, `solubility`, and `success_rate`.
- Relative drop is computed as `(clean_metric - perturbed_metric) / clean_metric * 100%` on higher-is-better transformed metrics.

## RFdiffusion Post-processing
- The pipeline first checks whether RFdiffusion candidates are backbone-only.
- If backbone-only PPDbench candidates are present, the intended path is `RFdiffusion backbone -> ProteinMPNN sequence recovery -> structure recovery / fallback peptide record` fully under this baseline directory.
- On the current machine, no direct PPDbench RFdiffusion peptide inputs were located; the scaffolded recovery output is recorded in `raw_results/rfdiffusion_mpnn_sequences.csv`.

## Intersection-only Principle
- `all_methods_robustness_summary_all_available.csv` keeps all methods with any valid results.
- `all_methods_robustness_summary_intersection.csv` keeps only targets shared by all four methods. If some methods are missing entirely on this machine, this table can be empty and the reason is preserved in logs and notes.

## Run Examples
```bash
python baseline/scripts/run_baseline_robustness.py --methods all --build-index-only
python baseline/scripts/run_baseline_robustness.py --methods proteingenerator --skip-existing
python baseline/scripts/plot_robustness_comparison.py
python /root/autodl-tmp/Peptide_3D/results/5_robustness/baseline/scripts/plot_figure5_robustness_comparison_updated.py
```

## Figure 5 (updated) — e–g panels (2026-03)

The legacy `scripts/plot_robustness_comparison.py` Figure 5 placed three per-perturbation affinity line plots in panels **e–g**, which largely **duplicated panel b** (global normalized affinity curves) and crowded the layout.

The **updated** script `scripts/plot_figure5_robustness_comparison_updated.py` keeps **a–d** in the same spirit (schematic, affinity curves, multi-metric drop heatmap, summary metrics) but **replaces e–g** so they complement **b** instead of repeating it:

| Panel | Role | Relation to **b** |
|-------|------|---------------------|
| **e** | Grouped bars of **relative affinity drop (%)** at **two representative strengths** per perturbation family: structure_missing & sequence_trunc at **20% / 40%**, pocket_noise at **1.0 Å / 2.0 Å** (moderate vs stronger noise). | **b** shows continuous normalized affinity vs strength; **e** isolates two operating points for direct cross-method comparison of degradation magnitude. |
| **f** | **Normalized success rate** vs perturbation strength, **same 3× small-multiple layout and method colors as b**. | **b** = binding (affinity) trajectory; **f** = success-rate trajectory on identical axes conventions → readers compare binding vs holistic success under the same perturbations. |
| **g** | **Affinity distribution** (violin + median) under **pocket noise 1.0 Å** per method, using **sample-level** CSVs filtered to **intersection targets** when `tables/intersection_targets.csv` is present. | **b** is mean-over-cohort curves; **g** exposes **dispersion / tails** at one representative noise level. |

Outputs are written only under this directory: `figures/Figure_5_robustness_comparison_updated.png|.pdf`, `figures/Figure_5_robustness_comparison_updated_caption.txt`, `tables/figure5_panel_E_representation_conditions.csv`, and `logs/figure5_robustness_updated_run.log`. The original `Figure_5_robustness_comparison.*` in `results/5_robustness/figures/` is **not** overwritten.

**Data choice:** the updated figure uses **intersection-only** merged tables (`all_methods_condition_curves_intersection.csv`, `all_methods_robustness_summary_intersection.csv`) for panels **a–f** and **h**. Panel **g** reads per-method `raw_results/<method>/samples_pocket_noise_lvl1p0_r0.csv` plus **Ours** `results/5_robustness/tables/samples_pocket_noise_lvl1p0_r0.csv` (read-only path outside `baseline/`), then restricts rows to intersection targets when available.

## Directory Layout
```text
repos/      official repositories and provenance snapshots
configs/    thresholds and runtime configs
scripts/    baseline pipeline and plotting entrypoints
logs/       stepwise execution logs
cache/      cleaned peptides, perturbed receptors, HDOCK/FoldX workdirs
raw_results/ sample-level per-method outputs and RFdiffusion sequence recovery tables
tables/     input indices, aggregates, summaries, merged all-available/intersection tables
metrics/    extra metric exports
figures/    comparison figure outputs and caption drafts
cases/      representative-case exports
tmp/        all temporary runtime files forced under baseline/
```

