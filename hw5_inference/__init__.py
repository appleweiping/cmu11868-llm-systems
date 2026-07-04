"""CMU 11-868 Assignment 6 — Advanced Training & Inference Systems.

The official assignment fine-tunes LLaMA-2-7B with DeepSpeed-ZeRO + LoRA and
serves it with SGLang (RadixAttention KV-cache reuse).  Both require multiple
16GB+ GPUs, so on this CPU build we implement and verify the *core systems
techniques* those frameworks are built on, at CPU scale, with real numbers:

* :mod:`hw5_inference.kv_cache` — incremental decoding with a key/value cache
  (the reuse RadixAttention generalises), verified bit-for-bit against full
  recomputation, with a measured per-token FLOP/latency reduction.
* :mod:`hw5_inference.quantization` — int8 (and int4) symmetric weight
  quantization with de-quantization, measured reconstruction error and the real
  4x/8x memory reduction, plus end-to-end LM-logit error.
* :mod:`hw5_inference.lora` — Low-Rank Adaptation: freeze the base weight, train
  a rank-``r`` update ``B @ A``; measured trainable-parameter reduction and a
  real fine-tuning run that lowers the loss.

The GPU-only pieces (7B-scale ZeRO training, SGLang server throughput) are
documented as partial in the README.
"""
