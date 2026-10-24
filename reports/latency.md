# Verb-model latency (batch 1)

Measured: 2026-10-24 05:53, device: `NVIDIA GeForce RTX 4060`, img_size 224x224, variant `tsn_avg`, 5 warmup + 30 timed forwards per cell.
Command: `python -m scripts.latency_bench --device cuda --img-size 224`

Note: measured with variant=tsn_avg, whose backbone runs on ALL T frames. single_frame runs the backbone on ONE frame, so its cost is close to the backbone cost at one frame plus a linear head (not measured separately here).

| backbone | T | precision | mean ms | min ms | windows/s |
|---|---:|---|---:|---:|---:|
| resnet18 | 4 | fp32 | 5.12 | 2.69 | 195.43 |
| resnet18 | 4 | fp16 | 3.47 | 2.28 | 288.36 |
| resnet18 | 8 | fp32 | 5.23 | 4.50 | 191.18 |
| resnet18 | 8 | fp16 | 3.08 | 2.63 | 324.43 |
| resnet18 | 16 | fp32 | 9.45 | 8.68 | 105.80 |
| resnet18 | 16 | fp16 | 5.66 | 5.23 | 176.80 |
| resnet34 | 4 | fp32 | 4.83 | 4.31 | 207.13 |
| resnet34 | 4 | fp16 | 4.03 | 3.47 | 248.20 |
| resnet34 | 8 | fp32 | 8.45 | 7.91 | 118.38 |
| resnet34 | 8 | fp16 | 5.55 | 4.69 | 180.06 |
| resnet34 | 16 | fp32 | 15.84 | 14.80 | 63.13 |
| resnet34 | 16 | fp16 | 9.62 | 8.82 | 103.96 |
| mobilenet_v2 | 4 | fp32 | 4.79 | 2.98 | 208.78 |
| mobilenet_v2 | 4 | fp16 | 4.95 | 3.70 | 201.84 |
| mobilenet_v2 | 8 | fp32 | 5.64 | 4.22 | 177.35 |
| mobilenet_v2 | 8 | fp16 | 5.83 | 3.85 | 171.65 |
| mobilenet_v2 | 16 | fp32 | 11.02 | 10.30 | 90.72 |
| mobilenet_v2 | 16 | fp16 | 5.27 | 4.66 | 189.91 |

GPU: NVIDIA GeForce RTX 4060 (this run WAS on cuda).
