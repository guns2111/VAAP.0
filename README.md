# VAAP.0
# Voice-Aware Adaptive Precision: Using Behavioral Drift Detection to Recover Quantization Loss in Large Language Models

**Alexander Lee Young**
Trinidad and Tobago

**March 2026**

---

## Abstract

We propose Voice-Aware Adaptive Precision (VAAP), a method for dynamically recovering quality loss in aggressively quantized large language models. Unlike existing mixed-precision approaches that determine layer precision statically through mathematical sensitivity analysis, VAAP uses lightweight probe networks trained on the target model's full-precision intermediate activations to detect behavioral drift during inference. When a probe identifies that a quantized layer's output diverges from the expected activation signature — indicating that the model is no longer "sounding like itself" — the system triggers selective full-precision recomputation for that layer only. This enables deployment of very large models (100B+ parameters) on memory-constrained hardware by running most layers at aggressive quantization (Q2/Q3) while dynamically recovering precision only where it matters. We outline the architecture, discuss computational trade-offs, and propose a prototype implementation path using commodity edge AI hardware.

## 1. Introduction

Deploying large language models on edge devices requires aggressive quantization. A 405B parameter model at 16-bit precision requires ~810GB of memory; at Q2 quantization, this compresses to ~100GB, approaching the capacity of high-end edge accelerators such as the NVIDIA Jetson AGX Orin (64GB unified memory + NVMe SSD offloading).

However, aggressive quantization introduces silent errors. Each weight is rounded from one of 65,536 possible values (FP16) to as few as 4 (Q2). These rounding errors are not uniformly distributed across layers — some layers tolerate aggressive quantization while others produce significantly degraded activations that propagate forward, corrupting downstream reasoning.

Current approaches to this problem fall into two categories:

**Static mixed-precision quantization** (GPTQ, AWQ, QuIP#) analyzes layer sensitivity at quantization time and assigns higher precision to layers with greater measured weight perturbation. This is effective but fixed — precision allocation cannot adapt to input-dependent variations in quantization error.

**Residual quantization** stores correction codebooks alongside quantized weights, enabling partial precision recovery at the cost of additional memory and a fixed computational overhead applied uniformly regardless of whether correction is needed.

We propose a third approach: **dynamic behavioral monitoring**, where a small auxiliary model evaluates whether each layer's quantized output is consistent with the target model's characteristic activation patterns, triggering precision recovery only when and where drift is detected.

## 2. Key Insight: Voice vs. Weight Error

Existing quantization error metrics operate at the weight level — measuring the mathematical distance between full-precision and quantized weight tensors (e.g., mean squared error, Hessian-weighted error). These metrics are necessary but insufficient. They answer the question: "How far is this weight from its true value?" They do not answer: "Does this error actually change what the model says?"

Many weight-level errors are functionally silent — absorbed by subsequent layers, cancelled by residual connections, or irrelevant to the specific input being processed. Conversely, small weight errors in critical positions can cascade through the network and produce qualitatively different outputs.

VAAP reframes quantization error detection from weight-level analysis to **behavioral analysis**: does the model still sound like itself? This is operationalized by training a small monitor on the target model's full-precision activation patterns and using it as a continuous quality signal during quantized inference.

The distinction is analogous to audio compression. Bitrate analysis tells you how much information was discarded. A trained listener tells you whether the music still sounds right. VAAP is the trained listener.

## 3. Architecture

### 3.1 Components

The system consists of three components:

**The Target Model (T):** The large model being deployed (e.g., 405B parameters), quantized aggressively (Q2 or Q3) for primary inference. Layers whose weights can be stored in fast unified memory reside there; overflow layers are stored on NVMe SSD and loaded on demand.

**The Monitor Model (M):** A set of lightweight probe networks — one per monitored layer — each consisting of 2-3 linear layers trained on the target model's full-precision intermediate activations. Each probe M_l takes the quantized activation at layer l and outputs a scalar drift score. The probes are small (tens of thousands of parameters each), reside entirely in fast memory, and add negligible computational overhead.

**The Precision Cache (C):** The complete target model's weights stored at full precision (FP16) on NVMe SSD. Because drift is input-dependent and any monitored layer may require correction on any given token, the full-precision copy of all monitored layers must be available. When the monitor detects drift at a specific layer, the full-precision weights for that layer are loaded into memory and the forward pass for that layer is recomputed.

### 3.2 Inference Pipeline

For each input token, the forward pass proceeds layer by layer:

1. At layer l, compute the quantized activation: a_l^q = Layer_l^{Q2}(a_{l-1}).
2. If layer l is monitored, pass a_l^q to monitor M, which outputs a drift score d_l = M_l(a_l^q), where M_l is the probe trained for layer l.
3. If d_l > θ_l (layer-specific threshold), load layer l's full-precision weights from C, recompute: a_l^{fp} = Layer_l^{FP16}(a_{l-1}), and pass a_l^{fp} to layer l+1.
4. If d_l ≤ θ_l, or if layer l is not monitored, pass a_l^q to layer l+1 unchanged.
5. Repeat until the final layer produces the output logits.

### 3.3 Monitor Training

The monitor probes are trained offline using the following procedure:

1. Run a representative corpus through the target model at full precision, recording intermediate activations at each monitored layer l: {a_l^{fp}(x_i)} for inputs x_i.
2. Run the same corpus through the quantized model, recording the corresponding activations: {a_l^q(x_i)}.
3. For each layer l and input x_i, compute the activation divergence: δ_l(x_i) = ||a_l^{fp}(x_i) - a_l^q(x_i)||_2 / ||a_l^{fp}(x_i)||_2 (normalized L2 distance).
4. Train probe M_l as a regression model: given a_l^q as input, predict δ_l. The probe learns which patterns of quantized activations correlate with high divergence from the full-precision reference.

At inference, the drift score d_l = M_l(a_l^q) is the probe's predicted divergence. The threshold θ_l is set per-layer as a percentile of the training-set divergence distribution — for example, θ_l at the 90th percentile means the system triggers recomputation only for activations estimated to be in the worst 10% of quantization error for that layer.

This is the critical trick: at inference time, the full-precision activations are unavailable. The probe must generalize from training examples to detect drift on novel inputs using only the quantized activation as input.

### 3.4 Layer Selection

Not all layers require monitoring. Empirically, quantization sensitivity clusters in specific architectural positions:

- **Attention layers** (especially early and late) are typically more sensitive than feed-forward layers.
- **The first and last 10-15% of layers** in transformer architectures show disproportionate sensitivity.
- **Layers immediately following residual connections** can amplify or absorb errors depending on the residual stream magnitude.

A practical deployment monitors 20-30% of layers, selected by offline sensitivity analysis. This reduces the monitor's overhead proportionally.

## 4. Computational Analysis

### 4.1 Overhead Budget

Let L = total layers, L_m = monitored layers, and p = probability of drift detection at a monitored layer.

**Monitor overhead per token:** L_m forward passes through small probe networks. Each probe is 2-3 linear layers with dimensions matching the model's hidden size (e.g., 8192 → 1024 → 1 for a 70B-class model). With L_m = 20 monitored layers, total probe computation is on the order of millions of multiply-accumulate operations per token — less than 0.01% overhead relative to the target model's forward pass. The monitor cost is effectively free.

**Recomputation overhead per token:** p × L_m full-precision layer recomputations. A 405B model with ~126 layers has approximately 3.2B parameters per layer, requiring ~6.4GB at FP16. Loading from NVMe SSD at 4-7 GB/s, each layer reload takes ~1-1.6s. If p = 0.1 (drift detected at 10% of monitored layers), the expected recomputation cost is ~2 layer reloads per token, adding ~2-3s.

**Expected total overhead:** The monitor itself is computationally negligible. The dominant cost is SSD reads for layer recomputation. For a model generating tokens at 10s/token (SSD-offloaded 405B at Q2), recomputation adds 2-3s per token on average (assuming 10% trigger rate across 20 monitored layers), yielding ~20-30% slowdown in exchange for targeted precision recovery. For low-frequency inference (one heartbeat per day), this is irrelevant. For smaller models (70-110B) with ~80 layers at ~1.7GB per layer, reloads are sub-second and overhead drops to ~5-10%.

### 4.2 Comparison to Alternatives

| Approach | Memory | Speed | Quality |
|---|---|---|---|
| Full Q2 (baseline) | Lowest | Fastest | Lowest — uniform degradation |
| Static mixed-precision | Low-Medium | Fast | Good — but fixed allocation |
| Residual quantization | Medium | Moderate | Good — but uniform overhead |
| Full FP16 (if it fits) | Highest | Fastest | Best — but requires massive memory |
| VAAP (405B) | Low + SSD cache | ~20-30% slower | Adaptive — precision where needed |
| VAAP (70-110B) | Low + SSD cache | ~5-10% slower | Adaptive — precision where needed |

VAAP's advantage is most pronounced on memory-constrained edge hardware where the choice is between aggressive quantization and not running the model at all.

## 5. Connection to Existing Work

**Speculative Decoding** (Leviathan et al., 2023) uses a small model to draft tokens and a large model to verify them. VAAP inverts this: the large model runs continuously and the small model monitors for quality. Both exploit asymmetric model costs for efficiency.

**Early Exit / Adaptive Computation** (Graves, 2016; Schuster et al., 2022) allows models to skip layers when confidence is high. VAAP applies adaptive computation to *precision* rather than depth — layers are not skipped but may be recomputed at higher precision.

**Quantization-Aware Training (QAT)** trains models to be robust to quantization. VAAP is complementary: it can be applied post-hoc to any quantized model without retraining.

**SmoothQuant** (Xiao et al., 2023) migrates quantization difficulty from activations to weights via per-channel scaling. This reduces activation outliers that cause quantization error. VAAP addresses a different failure mode: input-dependent error spikes that occur even in well-smoothed models.

**AQLM** (Egiazarian et al., 2024) uses additive quantization with learned codebooks to achieve high compression with lower error than round-to-nearest methods. VAAP could be layered on top of AQLM or any other quantization scheme — it is agnostic to the quantization method and operates purely at the behavioral level.

**Linear probes for model internals** (Belinkov, 2022; Alain & Bengio, 2017) use small classifiers on intermediate representations to understand what information is encoded at each layer. VAAP repurposes this technique: instead of asking "what does this layer know?", the probe asks "is this layer behaving normally?"

## 6. Limitations and Open Questions

**Storage requirements.** The precision cache requires storing the full model at FP16 on SSD. For a 405B model, this is ~810GB. Combined with the quantized model in memory (~100GB at Q2), total storage approaches 1TB. NVMe SSDs of this capacity are widely available, but the cost is non-trivial for edge deployments. For smaller models (70-110B), the cache is 140-220GB, which is modest.

**Monitor generalization.** The monitor must detect drift on inputs outside its training distribution. If the monitor fails to flag genuine drift, quantization errors pass through silently. Calibration of the detection threshold θ requires careful validation.

**Layer recomputation cost.** Loading full-precision weights from SSD introduces latency that may be unacceptable for interactive applications. VAAP is best suited for batch or low-frequency inference (e.g., daily journal entries, document analysis) rather than real-time chat.

**Training data requirements.** The monitor requires paired full-precision and quantized activations from a representative corpus. For domain-specific deployments, this corpus must reflect the target domain.

**Cascading corrections.** Correcting one layer changes the input to subsequent layers. A corrected activation at layer l may cause the monitor to flag layer l+2 that would not have been flagged with the quantized activation from layer l. The system should propagate corrections forward rather than evaluating all layers simultaneously.

**KV cache invalidation.** In autoregressive generation, attention layers maintain a key-value cache across tokens. If a layer's activation is recomputed at full precision for token t, the KV cache entries for that layer at tokens 1 through t-1 were computed at quantized precision and are now inconsistent. A strict implementation would require recomputing all prior KV entries for that layer at full precision, which is expensive. A pragmatic approximation is to accept the mixed-precision KV cache, on the assumption that the corrected activation at the current token matters more than historical cache consistency. The impact of this approximation on output quality is an open empirical question.

## 7. Proposed Implementation

A minimal proof-of-concept implementation requires:

1. **Framework:** HuggingFace Transformers with PyTorch forward hooks for activation capture.
2. **Target model:** Any quantized model available via GPTQ, bitsandbytes, or llama.cpp.
3. **Monitor probes:** A small MLP (e.g., hidden_dim → 1024 → 256 → 1) per monitored transformer layer, trained on paired full-precision and quantized activation data.
4. **Precision cache:** The full model weights at FP16 stored on NVMe SSD (~810GB for a 405B model).
5. **Hardware:** A single GPU or unified-memory accelerator (e.g., NVIDIA Jetson AGX Orin 64GB) with NVMe SSD.

Estimated implementation effort: 200-300 lines of Python for the prototype, excluding model loading and quantization infrastructure provided by existing libraries.

The key experiments are:

1. **Probe accuracy:** Can the probes reliably predict activation divergence from quantized activations alone? Measured as correlation between predicted and actual divergence on a held-out set.
2. **Quality recovery:** Does selective recomputation improve perplexity and downstream task performance relative to uniform quantization? Measured across Q2, Q3, Q4 on standard benchmarks (WikiText, MMLU, HumanEval).
3. **Efficiency:** What is the actual recomputation trigger rate, and how does the speed-quality tradeoff compare to static mixed-precision baselines?

## 8. Conclusion

Voice-Aware Adaptive Precision reframes quantization error recovery as a behavioral monitoring problem rather than a mathematical optimization problem. By asking "does the model still sound like itself?" rather than "how far are the weights from their true values?", VAAP enables dynamic, input-dependent precision allocation that targets computational resources at the specific moments where quantization actually degrades output quality.

The approach is motivated by a practical observation from edge deployment: a 110B model at Q5 (fully in memory, high precision) consistently outperforms a 405B model at Q2 (partially on disk, low precision) — not because it knows fewer things, but because it can finish a thought clearly. VAAP aims to give the 405B model the same clarity, selectively, where it matters.

---

## References

Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. *ICLR Workshop*.

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*, 48(1), 207-219.

Dettmers, T., et al. (2022). GPT3.int8(): 8-bit matrix multiplication for transformers at scale. *NeurIPS*.

Egiazarian, V., et al. (2024). Extreme compression of large language models via additive quantization. *ICML*.

Frantar, E., et al. (2023). GPTQ: Accurate post-training quantization for generative pre-trained transformers. *ICLR*.

Graves, A. (2016). Adaptive computation time for recurrent neural networks. *arXiv:1603.08983*.

Leviathan, Y., et al. (2023). Fast inference from transformers via speculative decoding. *ICML*.

Lin, J., et al. (2024). AWQ: Activation-aware weight quantization for LLM compression and acceleration. *MLSys*.

Schuster, T., et al. (2022). Confident adaptive language modeling. *NeurIPS*.

Tseng, A., et al. (2024). QuIP#: Even better LLM quantization with Hadamard incoherence and lattice codebooks. *ICML*.

Xiao, G., et al. (2023). SmoothQuant: Accurate and efficient post-training quantization for large language models. *ICML*.

---

## Origin Note

This architecture was conceived during a late-night conversation about deploying large language models on a borrowed Jetson on a balcony in Trinidad. The first author was supposed to be sleeping. The insight — that behavioral drift from a LoRA-tuned monitor is a better error signal than weight-level perturbation analysis — emerged from direct experience with LoRA fine-tuning applied to a garden-tending AI system (TriniSeed), where a small model trained on its own outputs developed a measurable "voice" distinct from its base weights.

The idea that you could use one model's voice as a quality probe for another model's coherence is, in retrospect, obvious. Most good ideas are.

---

*Correspondence: alexleeyoung@pingtt.com*

*This paper describes a proposed architecture. No experimental results are reported. The author welcomes collaboration on implementation and evaluation.*
