# DNSMOS Checkpoints

The evaluator requires the following DNSMOS ONNX checkpoints:

- `sig_bak_ovr.onnx`
- `model_v8.onnx`

These checkpoint files are not included in this repository.

Please obtain the required DNSMOS files from the official Microsoft DNS Challenge
repository and provide their local paths when running `evaluate.py`.

Example:

```bash
--dns_primary_model ./dnsmos/sig_bak_ovr.onnx \
--dns_p808_model ./dnsmos/model_v8.onnx
