# Model Verification Report

Generated (UTC): 2026-11-19T04:15:58.493663+00:00

## Status of each verification step

| Step | Status |
|---|---|
| file inspected | YES |
| architecture instantiated | YES |
| strict load ok | YES |
| preprocessing verified against training | NO / NOT VERIFIED |
| class mapping verified | NO / NOT VERIFIED |
| real inference completed | NO / NOT VERIFIED |
| output geometry verified | NO / NOT VERIFIED |

## Checkpoint
- File: `best_unet_brats.pth`
- Size: 97935923 bytes
- SHA-256: `a15086d57e073b77ef9e14221f9f614be4f06b76a4a8145e803c2f2935f5e32b`
- Container key: `None`
- Tensors: 278
- Inferred input channels: 4
- Inferred output classes: 4
- Top-level keys: ['encoder.conv1.weight', 'encoder.bn1.weight', 'encoder.bn1.bias', 'encoder.bn1.running_mean', 'encoder.bn1.running_var', 'encoder.bn1.num_batches_tracked', 'encoder.layer1.0.conv1.weight', 'encoder.layer1.0.bn1.weight', 'encoder.layer1.0.bn1.bias', 'encoder.layer1.0.bn1.running_mean', 'encoder.layer1.0.bn1.running_var', 'encoder.layer1.0.bn1.num_batches_tracked', 'encoder.layer1.0.conv2.weight', 'encoder.layer1.0.bn2.weight', 'encoder.layer1.0.bn2.bias', 'encoder.layer1.0.bn2.running_mean', 'encoder.layer1.0.bn2.running_var', 'encoder.layer1.0.bn2.num_batches_tracked', 'encoder.layer1.1.conv1.weight', 'encoder.layer1.1.bn1.weight', 'encoder.layer1.1.bn1.bias', 'encoder.layer1.1.bn1.running_mean', 'encoder.layer1.1.bn1.running_var', 'encoder.layer1.1.bn1.num_batches_tracked', 'encoder.layer1.1.conv2.weight', 'encoder.layer1.1.bn2.weight', 'encoder.layer1.1.bn2.bias', 'encoder.layer1.1.bn2.running_mean', 'encoder.layer1.1.bn2.running_var', 'encoder.layer1.1.bn2.num_batches_tracked', 'encoder.layer1.2.conv1.weight', 'encoder.layer1.2.bn1.weight', 'encoder.layer1.2.bn1.bias', 'encoder.layer1.2.bn1.running_mean', 'encoder.layer1.2.bn1.running_var', 'encoder.layer1.2.bn1.num_batches_tracked', 'encoder.layer1.2.conv2.weight', 'encoder.layer1.2.bn2.weight', 'encoder.layer1.2.bn2.bias', 'encoder.layer1.2.bn2.running_mean', 'encoder.layer1.2.bn2.running_var', 'encoder.layer1.2.bn2.num_batches_tracked', 'encoder.layer2.0.conv1.weight', 'encoder.layer2.0.bn1.weight', 'encoder.layer2.0.bn1.bias', 'encoder.layer2.0.bn1.running_mean', 'encoder.layer2.0.bn1.running_var', 'encoder.layer2.0.bn1.num_batches_tracked', 'encoder.layer2.0.conv2.weight', 'encoder.layer2.0.bn2.weight']
- Embedded metadata: {}

## Architecture
```
{
  "name": "smp.Unet",
  "encoder_name": "resnet34",
  "encoder_weights": null,
  "in_channels": 4,
  "classes": 4,
  "activation": null
}
```
- Parameters: {'total': 24439940, 'trainable': 24439940}

## Compatibility
```
{
  "missing_keys": [],
  "n_missing": 0,
  "unexpected_keys": [],
  "n_unexpected": 0,
  "shape_mismatches": [],
  "n_shape_mismatches": 0
}
```

## Environment
```
{
  "python": "3.11.9",
  "torch": "2.15.0+cpu",
  "smp": "0.6.0",
  "os": "Windows-10-10.0.26300-SP0",
  "cuda_available": false
}
```

## Interpretation
A passing strict load only shows the weights fit the configured architecture. It does NOT show that the preprocessing or label mapping match training, nor that predictions are accurate. Those require the training code (not supplied) and evaluation on real BraTS data with an independent patient-wise test set.