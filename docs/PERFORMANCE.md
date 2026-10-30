# Inference performance

## CUDA Graph replay

CUDA FP32 inference automatically uses CUDA Graph replay when the device supports capture. The checkpoint, parameter values, and FP32 computations are unchanged. Graphs are captured lazily per `(batch size, token bucket)` and replayed against reusable input buffers. Text is padded to 16, 32, 64, or 128 tokens. Up to eight recently used shapes are cached; an unsupported capture disables graph replay and falls back to eager inference. CPU and BF16/FP16 use eager inference. Set `SFC128_CUDA_GRAPHS=0` or pass `--eager` to select eager CUDA inference.

The first request using a new shape pays the graph-capture cost. Benchmark reports include that capture time separately, while p50/p95 latency includes steady-state runs after warmup.

## RTX 3060 comparison

Measured on Windows 11 with Python 3.10, PyTorch 2.5.1+cu121, an NVIDIA GeForce RTX 3060, FP32, and a 28-token English input. The eager path loaded the original trainer export at `D:\path\to\SemanticFloat\sfc_emotion_v2_run\SFC128_Emotion`; the graph path loaded this repository's attached export. The checkpoint and tokenizer SHA256 hashes match. Each run used five warmup calls and 30 timed calls through the same benchmark, including tokenization, batching, model inference, softmax, and CPU result conversion. Graph capture happened during warmup.

| Batch size | Original eager p50 | Graph p50 | Original eager p95 | Graph p95 | Throughput, eager → graph |
|---:|---:|---:|---:|---:|---:|
| 1 | 166.213 ms | **10.323 ms** | 192.083 ms | **10.756 ms** | 6.016 → **96.867 examples/s** |
| 16 | 144.971 ms | **14.005 ms** | 179.781 ms | **15.666 ms** | 110.367 → **1,142.437 examples/s** |

These are local results for the tested GPU, driver, and software build. A fresh engine captured its first batch-1/32-token graph in **695 ms**; the full first request took 707 ms. Graph replay increased measured peak CUDA allocation from about 13.2 MB to 20.6 MB for these shapes. Other input lengths/batch shapes incur their own first-capture cost and may have different latency.

## Prediction parity

The eager FP32 original export and graph FP32 repository export were compared on all **1,998 cached validation examples**. Every cached token sequence round-tripped exactly through the paired tokenizer. The predicted class matched on every example, and the maximum absolute probability difference was **0.0**. Both paths measured 86.7868% validation accuracy and 0.852552 macro-F1 on this cache.

The saved training metrics report 86.9369% validation accuracy and 0.853272 macro-F1. The training run used automatic precision and the final evaluation log explicitly reports BF16, while this parity comparison held FP32 constant to isolate the serving optimization. Therefore the saved mixed-precision result and FP32 parity score are different measurements; the key comparison here is that eager FP32 and CUDA Graph FP32 agree exactly.

## Why graph replay helps

The model's 128 blocks are sequential and have independent learned weights, so combining blocks or sharing weights would change the trained model. A profiler showed many small launches for REN-T group reductions and the Floating gate, plus repeated reshape/cast/pointwise work. The depthwise convolution was a small portion of measured GPU operator time. CUDA Graph replay reduces host launch overhead while preserving those operations and their ordering. A fused REN-T or Floating-gate kernel may improve the device-side work further, but it would require a custom kernel implementation and its own numerical and performance validation.

`torch.compile` was also tried on the installed PyTorch 2.5.1 environment; Inductor could not compile because no working Triton installation was available. It is not required for the graph path.

