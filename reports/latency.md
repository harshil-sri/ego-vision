# Verb-model latency (batch 1)

Measured: 2026-10-23 23:45, device: `CPU (AMD Ryzen 7 7700X 8-Core Processor)`, img_size 96x96, variant `tsn_avg`, 5 warmup + 30 timed forwards per cell.
Command: `python -m scripts.latency_bench --device cpu --img-size 96`

Note: measured with variant=tsn_avg, whose backbone runs on ALL T frames. single_frame runs the backbone on ONE frame, so its cost is close to the backbone cost at one frame plus a linear head (not measured separately here).

| backbone | T | precision | mean ms | min ms | windows/s |
|---|---:|---|---:|---:|---:|
| resnet18 | 4 | fp32 | 10.78 | 9.28 | 92.79 |
| resnet18 | 8 | fp32 | 17.44 | 15.62 | 57.35 |
| resnet18 | 16 | fp32 | 28.97 | 25.83 | 34.52 |
| resnet34 | 4 | fp32 | 21.00 | 18.65 | 47.61 |
| resnet34 | 8 | fp32 | 29.75 | 28.29 | 33.61 |
| resnet34 | 16 | fp32 | 50.05 | 45.25 | 19.98 |
| mobilenet_v2 | 4 | fp32 | 9.47 | 7.57 | 105.57 |
| mobilenet_v2 | 8 | fp32 | 14.48 | 11.01 | 69.07 |
| mobilenet_v2 | 16 | fp32 | 35.92 | 32.19 | 27.84 |

Skipped cells:
- resnet18 T=4 fp16: skipped on CPU: fp16 gives no CPU speedup and several ops are unsupported
- resnet18 T=8 fp16: skipped on CPU: fp16 gives no CPU speedup and several ops are unsupported
- resnet18 T=16 fp16: skipped on CPU: fp16 gives no CPU speedup and several ops are unsupported
- resnet34 T=4 fp16: skipped on CPU: fp16 gives no CPU speedup and several ops are unsupported
- resnet34 T=8 fp16: skipped on CPU: fp16 gives no CPU speedup and several ops are unsupported
- resnet34 T=16 fp16: skipped on CPU: fp16 gives no CPU speedup and several ops are unsupported
- mobilenet_v2 T=4 fp16: skipped on CPU: fp16 gives no CPU speedup and several ops are unsupported
- mobilenet_v2 T=8 fp16: skipped on CPU: fp16 gives no CPU speedup and several ops are unsupported
- mobilenet_v2 T=16 fp16: skipped on CPU: fp16 gives no CPU speedup and several ops are unsupported

RTX 4060: NOT RUN (owner asleep; run scripts/latency_bench.py --device cuda --img-size 224 in the morning)
