Finetuned SmolLM3 to optimize the GPU memory usage, for a 3070 RTX Laptop GPU.


### Memory Trade-offs:

- 4-bit quantization: 12GB → 4GB (quality: ~98%)
- LoRA: Train 0.5% of params (quality: 95-98%)
- Gradient checkpointing: Save 40% VRAM, cost 20% speed
- fp16: Use 50% memory vs fp32


### Effective Batch Size:

Real batch size = per_device_batch_size × gradient_accumulation_steps
                = 4 × 2 = 8

Higher effective batch = more stable training
Lower per_device_batch = less VRAM usage


### Training Speed Impact:

- fp16 vs fp32: ~2x faster
- bf16 vs fp16: ~same speed (better stability)
- gradient_checkpointing=False vs True: ~25% faster
- packing=True: 10-20% faster (but can cause issues)
- max_length=256 vs 512: ~40% faster
- batch_size=4 vs 2: ~30% fewer iterations
- dataloader_num_workers=4 (Linux): ~15% faster overall
  

### Performance Summary:

Current config on RTX 3070:
- Speed: ~3.2s/iteration
- Total time: ~53 minutes for 1000 steps
- VRAM usage: ~6-7GB
- Stable training without OOM

 
