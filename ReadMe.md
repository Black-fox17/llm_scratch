# Mini-LLM Lab

A modularized repository for building and exploring Large Language Models from scratch.

## Project Structure

- `data/`: Datasets and preprocessing scripts.
- `tokenization/`: Byte-pair encoding, sentencepiece, etc.
- `embeddings/`: Word and positional embeddings.
- `attention/`: Multi-head attention, flash attention, etc.
- `transformer/`: Block and full model definitions.
- `sampling/`: Greedy search, beam search, top-k/top-p.
- `kv_cache/`: Key-value caching for efficient inference.
- `long_context/`: RoPE, Alibi, Sliding Window Attention.
- `moe/`: Mixture of Experts implementation.
- `norms/`: LayerNorm, RMSNorm.
- `objectives/`: Cross-entropy, contrastive loss.
- `scaling/`: Parallelism and distributed training.
- `quant/`: Post-training quantization, QLoRA.
- `infra/`: Logging, training loops, accelerators.
- `utils/`: Visualization and metrics.

## Getting Started

1. Prepare your data in `data/`.
2. Implement your tokenizer in `tokenization/`.
3. Build the transformer blocks in `transformer/`.
