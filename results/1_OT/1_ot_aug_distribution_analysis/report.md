# OT Self-Distillation Augmentation Analysis

## Dataset scale

| Metric | Value |
|---|---:|
| n_original | 9032 |
| n_augmented | 56172 |
| scale_ratio | 6.2192 |
| n_targets_original | 9032 |
| n_targets_augmented | 8630 |

## Peptide length distribution

### Original summary

| Metric | Value |
|---|---:|
| n | 9032 |
| mean | 14.5389 |
| std | 11.4200 |
| min | 3.0000 |
| p25 | 7.0000 |
| median | 10.0000 |
| p75 | 18.0000 |
| max | 50.0000 |

### Augmented summary

| Metric | Value |
|---|---:|
| n | 56172 |
| mean | 14.8082 |
| std | 10.2393 |
| min | 2.0000 |
| p25 | 8.0000 |
| median | 11.0000 |
| p75 | 18.0000 |
| max | 50.0000 |

### Comparison metrics

| Metric | Value |
|---|---:|
| wasserstein_1d | 1.1488 |
| ks_statistic | 0.1092 |
| mean_shift | 0.2693 |

## Amino-acid composition

### Original summary

| Metric | Value |
|---|---:|
| A | 0.0806 |
| C | 0.0190 |
| D | 0.0512 |
| E | 0.0691 |
| F | 0.0418 |
| G | 0.0602 |
| H | 0.0213 |
| I | 0.0486 |
| K | 0.0645 |
| L | 0.0987 |
| M | 0.0206 |
| N | 0.0372 |
| P | 0.0622 |
| Q | 0.0401 |
| R | 0.0693 |
| S | 0.0645 |
| T | 0.0479 |
| V | 0.0550 |
| W | 0.0168 |
| Y | 0.0316 |

### Augmented summary

| Metric | Value |
|---|---:|
| A | 0.0742 |
| C | 0.0189 |
| D | 0.0492 |
| E | 0.0671 |
| F | 0.0434 |
| G | 0.0524 |
| H | 0.0228 |
| I | 0.0497 |
| K | 0.0648 |
| L | 0.1024 |
| M | 0.0206 |
| N | 0.0382 |
| P | 0.0620 |
| Q | 0.0426 |
| R | 0.0744 |
| S | 0.0684 |
| T | 0.0457 |
| V | 0.0532 |
| W | 0.0178 |
| Y | 0.0322 |

### Comparison metrics

| Metric | Value |
|---|---:|
| js_divergence | 0.0006 |
| l1_distance | 0.0449 |

## Secondary-structure composition

### Original summary

| Metric | Value |
|---|---:|
| C | 0.8000 |
| E | 0.0171 |
| H | 0.1829 |

### Augmented summary

| Metric | Value |
|---|---:|
| C | 0.7641 |
| E | 0.0181 |
| H | 0.2178 |

### Comparison metrics

| Metric | Value |
|---|---:|
| js_divergence | 0.0014 |
| l1_distance | 0.0716 |

## Affinity label distribution

### Original summary

| Metric | Value |
|---|---:|
| n | 8573 |
| mean | -873.5653 |
| std | 6043.8950 |
| min | -99548.3900 |
| p25 | -393.0400 |
| median | -259.6900 |
| p75 | -187.4400 |
| max | -23.4300 |

### Augmented summary

| Metric | Value |
|---|---:|
| n | 53482 |
| mean | -661.1928 |
| std | 1455.1808 |
| min | -14874.1700 |
| p25 | -395.9000 |
| median | -340.6400 |
| p75 | -299.1200 |
| max | -61.2500 |

### Comparison metrics

| Metric | Value |
|---|---:|
| wasserstein_1d | 701.2749 |
| ks_statistic | 0.4062 |
| mean_shift | 212.3725 |

## Stability label distribution

### Original summary

| Metric | Value |
|---|---:|
| n | 9032 |
| mean | -13.2324 |
| std | 14.0677 |
| min | -248.3080 |
| p25 | -16.4556 |
| median | -8.7541 |
| p75 | -4.8867 |
| max | 9.6304 |

### Augmented summary

| Metric | Value |
|---|---:|
| n | 55382 |
| mean | -29.2439 |
| std | 11.6213 |
| min | -266.8650 |
| p25 | -36.9894 |
| median | -28.6418 |
| p75 | -20.7042 |
| max | 9.5576 |

### Comparison metrics

| Metric | Value |
|---|---:|
| wasserstein_1d | 16.5408 |
| ks_statistic | 0.6118 |
| mean_shift | -16.0115 |

## Solubility label distribution

### Original summary

| Metric | Value |
|---|---:|
| n | 2128 |
| mean | 0.7029 |
| std | 0.1293 |
| min | 0.3160 |
| p25 | 0.6130 |
| median | 0.7110 |
| p75 | 0.7983 |
| max | 1.0000 |

### Augmented summary

| Metric | Value |
|---|---:|
| n | 55382 |
| mean | 0.7246 |
| std | 0.1223 |
| min | 0.3620 |
| p25 | 0.6160 |
| median | 0.7200 |
| p75 | 0.8240 |
| max | 1.0000 |

### Comparison metrics

| Metric | Value |
|---|---:|
| wasserstein_1d | 0.0225 |
| ks_statistic | 0.0819 |
| mean_shift | 0.0217 |

## Notes

- Target family distribution was not analyzed because no target-family annotation file was provided.