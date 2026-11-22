# Demonstration checklist (status at build time)
| Scenario | Synthetic data | Real MRI | Real checkpoint | BraTS reference mask | Notes |
|---|---|---|---|---|---|
| 1 Upload & validation (accept, missing, corrupt, mismatch) | VERIFIED (pytest, live HTTP) | NOT RUN | n/a | n/a | |
| 2 AI segmentation + viewer | VERIFIED pipeline with random weights (pytest) and with the **real checkpoint on synthetic noise volumes** (live HTTP) – only proves the code runs | NOT RUN | runs; predictions on real MRI NOT RUN | n/a | Slice viewer endpoint tested; visual browser check NOT done |
| 3 Radiomics (features, settings, CSV/Excel) | VERIFIED (pytest) | NOT RUN | NOT RUN | NOT RUN | |
| 4 Evaluation WT/TC/ET | VERIFIED (pytest, synthetic GT) | NOT RUN | NOT RUN | NOT RUN | |
| 5 PDF + history + delete | VERIFIED (pytest, live HTTP) | NOT RUN | NOT RUN | NOT RUN | |
Complete the "NOT RUN" cells on your machine with BraTS cases (see README checklist) and record evidence (screenshots, exported JSON).
