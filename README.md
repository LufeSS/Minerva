# Minerva — Budget LLM (Minerva-B & Minerva-X)

**Minerva** is a **budget-LLM research effort**.  The goal is to prototype novel training tricks (kernel-space vocab heads, ACT, continuous latents, SF-SWA, etc.) and distillation recipes that make it possible to train a *useful 1-3 billion-parameter* decoder on no more than a single Azure T4 instance.  The project is organised into two complementary module families that can be trained and evaluated independently or mixed inside one model:

| Family | Folder | Purpose |
|--------|--------|---------|
| **Minerva-B** – *Baseline* | `src/minerva/` | A modern GPT-style decoder (LayerNorm+SwiGLU+linear-complexity attention) retained *verbatim* from the original Minerva repository and used as an anchor / control when evaluating Minerva-X ideas. |
| **Minerva-X** – *Experimental* | `src/minervax/` | A newer kernel–space variant which introduces Adaptive-Computation-Time (ACT), continuous latent feedback, Butterfly-Givens FFNs, and Rotary Positional Embeddings (RoPE). |

The repository additionally ships a **self-contained Kaggle script** (`kaggle_ksv_transformer_script.py`) that mirrors the library code in one file so the model can be trained inside Kaggle notebooks without installing the package.

---
## Why another transformer?

The goal is to **explore kernel–space vocabulary heads and softmax-free residual averaging** while keeping the compute footprint small enough to run on consumer GPUs.  In practice this means replacing the usual softmax output layer with a Mahalanobis distance head (`KSVHead`) and pushing most of the heavy lifting into *latent* space rather than logits.

Key ideas currently implemented:

* Nyström multi-head attention ⇒  \(\mathcal O(T)\) memory
* Butterfly-Givens single-layer FFNs – parameter efficient and highly parallel
* Extended arctan highways – learnable gates around *every* residual path
* ACT (Graves, 2016) with token-wise slope modulation – variable per-token depth
* Kernel-Space Vocabulary head – μ/σ² prediction + diagonal Mahalanobis logits
* Continuous latent autoregression – feed whitened latents back instead of ids
* K-future token prediction (default K = 3) for richer learning signals
* SF-SWA (stream-friendly stochastic weight averaging) for on-GPU averaging
* Rotary Positional Embeddings (RoPE) – unlimited context, no learned table

---
## Repository layout

```
ksv_transformer/              # this folder (root package)
│
├─ src/minerva/               # Minerva-B code (baseline)
│   └─ …                      # decoder, layers, data helpers
│
├─ src/minervax/              # Minerva-X code (experimental)
│   └─ model.py               # MinervaXTransformer
│
├─ kaggle_ksv_transformer_script.py   # one-file version for Kaggle kernels
└─ ... (training scripts, cached datasets, etc.)
```

---
## Quick start

```bash
pip install -r requirements.txt

python -m ksv.train \
  --epochs 3 \
  --batch_size 16 \
  --sample_prompt "history of"
```

To experiment with the RoPE/ACT variant:

```python
from minervax import MinervaXTransformer
model = MinervaXTransformer(vocab_size=2048)
logits = model(input_ids)
```

The trainer will gain a `--arch minervax` switch soon – for now import the
class manually or swap the import path.

---
### Roadmap

#### Phase 1 – Module development

* **Minerva-B (baseline)** – consolidate & refactor:
  * SwiGLU feed-forward
  * Highway residual between attention and FFN
  * Nyström linear-complexity attention
  * Sequential softmax residual

* **Minerva-X (experimental)** – new ideas:
  * Adaptive Computation Time (ACT)
  * Extended arctan-gated GLU block
  * Running softmax residual
  * Continuous latent feedback (kernel-space prediction)
  * K-future token head (multi-token prediction)
  * Streaming-friendly SWA (SF-SWA)
  * (Adversarial training – postponed for now)

#### Phase 2 – Integration & smoke-tests

1. Combine Minerva-B and Minerva-X layers into a single *configurable* architecture.
2. Train on **Wikipedia-Small** (≈20 M tokens) on a single Azure T4 GPU to check stability and speed.
3. Polish edge-cases, API, and generation helpers.

#### Phase 3 – Distillation & scaling-up

1. **Pre-train distillation** using [Llama-405B logits](https://huggingface.co/datasets/arcee-ai/LLama-405B-Logits).
2. **Fine-tune distillation** from:
   * Llama-2 70B-Instruct (STEM subset)
   * Llama-3.2 3B (coding subset)
   * Tool: [Logits-Based-Finetuning](https://github.com/dvlab-research/Logits-Based-Finetuning)
3. Target a **1-3 B parameter model** that *fits into a single T4* for both training and inference.
4. Explore hyper-parameter tuning, SF-SWA checkpoints, and quantisation.

---
## Citation

If you use Minerva-X or the KSV head in academic work please cite:

```text
@software{minerva_2025,
  author  = {LufeSS and contributors},
  title   = {Minerva: Budget LLM (Minerva-B & Minerva-X)},
  year    = {2025},
  url     = {https://github.com/LufeSS/Minerva}
}
```

---
## License

This project is released under the same **MIT License**. 
