# Patch Bank

This directory stores dynamically generated LoRA recovery adapters produced by the Autonomous LoRA Self-Healing Framework.

During anomaly recovery, validated LoRA patches may be saved here using:

```python
healed_model.save_pretrained("checkpoints/patch_bank/active_stable_patch")
```

Stored adapters can later be reloaded for analysis, benchmarking, or deployment.

## Notes

* Generated checkpoint files are excluded from version control.
* Large model artifacts should not be committed to the repository.
* This directory is intended for runtime-generated recovery patches.
