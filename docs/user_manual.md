# User manual
(Screenshots: add your own at the marked placeholders after running the app on your machine; none were captured in the build environment.)
1. **New analysis** – choose T1, T1ce, T2, FLAIR (+ optional ground truth), press *Upload & validate*. Errors (missing file, wrong type, corrupt file, shape/affine mismatch, bad GT labels) are listed. `[SCREENSHOT: upload page]`
2. **Results** – press *Run segmentation*; stages update live. Then use the viewer: modality, view (axial/coronal/sagittal), region (All/WT/TC/ET), source (predicted/ground truth), display mode, slice slider or mouse wheel, opacity. Record a review status. `[SCREENSHOT: viewer]`
3. **Radiomics** – choose source, regions, modalities → *Extract*. Filter/search the table, view the histogram, export CSV/Excel. If both predicted and ground-truth runs exist, a paired comparison appears.
4. **Evaluation** – only with a ground-truth upload; shows metrics, chart and policy. Without GT the page says accuracy cannot be calculated.
5. **Report** – *Generate new report* then *Download latest PDF*; also *Metadata JSON* and *Prediction NIfTI*.
6. **History** – reopen, download PDF, delete (confirmation dialog; removes DB rows and files).
