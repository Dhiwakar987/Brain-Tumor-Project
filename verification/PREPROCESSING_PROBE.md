# Preprocessing probe
Slices used: 40 · mask: labels [0, 1, 2, 3] used directly as class indices

| Preprocessing | Channel order | WT Dice | class1 | class2 | class3 |
|---|---|---|---|---|---|
| div255 | as stored (0,1,2,3) | 0.899 | 0.571 | 0.841 | 0.775 |
| zscore_nonzero_per_slice | as stored (0,1,2,3) | 0.717 | 0.130 | 0.268 | 0.413 |
| div_slice_max | as stored (0,1,2,3) | 0.899 | 0.571 | 0.841 | 0.775 |
| raw_0_255 (control, expected bad) | as stored (0,1,2,3) | 0.119 | 0.025 | 0.028 | 0.081 |
| div255 | (0, 1, 2, 3) | 0.899 | 0.571 | 0.841 | 0.775 |
| div255 | (1, 0, 2, 3) | 0.755 | 0.251 | 0.623 | 0.087 |
| div255 | (0, 2, 1, 3) | 0.736 | 0.026 | 0.279 | 0.200 |
| div255 | (0, 1, 3, 2) | 0.724 | 0.395 | 0.628 | 0.631 |
| div255 | (1, 2, 0, 3) | 0.610 | 0.012 | 0.232 | 0.092 |
| div255 | (0, 3, 2, 1) | 0.599 | 0.084 | 0.013 | 0.241 |

## Verdict
- Preprocessing: NOT DECIDED (best WT Dice 0.899, margin to runner-up 0.000). Do not configure from this probe.
- Channel order: CLEAR winner ((0, 1, 2, 3)), margin 0.143 in WT Dice

Confusion (rows = mask class 0-3, cols = predicted class 0-3) with best settings:
```
[[2248364      11    1701     171]
 [    544    6145     945     476]
 [   4184     231   31860     243]
 [    238     183     348    8356]]
```
Diagnostic on training-format slices (likely seen in training); not an accuracy estimate. Pick the config only if one candidate is clearly better; if several tie, the probe cannot decide.