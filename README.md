# Interpretability Analysis of Bilingual Brain-Informed Fine-Tuning in BERT

This repository contains the analysis code for the lab rotation **“Interpretability Analysis of Bilingual Brain-Informed Fine-Tuning in BERT.”** It investigates how bilingual, fMRI-guided fine-tuning changes the internal representations and causal computations of BERT-style masked language models.

The original brain-informed network architecture and fine-tuned weights are provided by the [denizenslab/brain-informed-fine-tuning](https://github.com/denizenslab/brain-informed-fine-tuning) project. 

## Overview

BERT is a Transformer encoder pretrained with a masked-language-model objective, producing deeply bidirectional contextual representations by conditioning each token on both its left and right context [Devlin et al., 2019](https://aclanthology.org/N19-1423/). In the original brain-informed fine-tuning framework, bilingual participants read naturalistic stories in English and Chinese while their BOLD responses were recorded. Token-level language-model representations are temporally aligned with fMRI responses using differentiable downsampling and haemodynamic delays. A voxelwise prediction head is trained to predict BOLD activity, and the resulting brain-encoding loss is backpropagated through the language model. Thus, the model is tuned to produce representations that better predict bilingual neural responses during naturalistic language comprehension [Negi et al., 2025](https://openreview.net/forum?id=JPogehP8By).

The aim of this repository is to ask a mechanistic question: **what changes inside the model after bilingual brain-informed supervision?** We compare fine-tuned variants with their corresponding pretrained base models across syntax-sensitive structure, representational geometry, language-specific output routing, factual retrieval, and causal pathways through layers and attention heads.

## Model variants

Within each model family, analyses compare four checkpoint conditions:

| Variant | Fine-tuning regime | Interpretation |
|---|---|---|
| **Base** | No brain-informed fine-tuning; pretrained checkpoint | Reference model for all comparisons. It establishes the original representational geometry, factual-retrieval behaviour, language-token probability mass, and causal attribution profile. |
| **Whole** | Fine-tuned to predict fMRI responses from **whole-brain voxels** | Broad brain supervision. The prediction target includes all available voxels rather than a functionally selected subset. |
| **Semantic** | Fine-tuned to predict fMRI responses from **semantically selective voxels** | Supervision is restricted to cortical voxels selected for semantic sensitivity, testing whether semantic brain signals specifically reshape language-model computation. |
| **Language** | Fine-tuned to predict fMRI responses from **language-selective voxels** | Supervision is restricted to language-selective cortical regions, testing whether language-network signals preferentially alter linguistic representations and language-specific routing. |

### Model families and training data

The original study fine-tunes both multilingual and monolingual BERT-family models with the same voxel-mask regimes:

- **mBERT**: multilingual BERT variants fine-tuned using fMRI collected while bilingual participants read either **English** or **Chinese** stories. This yields six brain-informed mBERT conditions: English-trained and Chinese-trained checkpoints for each of the Whole, Semantic, and Language masks.
- **BERT-en**: monolingual English BERT fine-tuned using the Whole, Semantic, and Language voxel masks.
- **BERT-zh**: monolingual Chinese BERT fine-tuned using the same three voxel-mask regimes.

All analyses operate on the hidden-state output of each Transformer block. Unless otherwise stated, each fine-tuned checkpoint is compared to the Base model from the same model family. The original fine-tuning work reports that bilingual brain-informed supervision can improve brain encoding and downstream NLP performance across within-language, cross-language, and, in multilingual models, and unseen-language settings [Negi et al., 2025](https://openreview.net/forum?id=JPogehP8By).

## Interpretability analyses

### 1. Structural probe analysis

`Probe_analysis.py` evaluates whether brain-informed fine-tuning changes how syntactic dependency structure is encoded across layers. Word-piece representations are mean-pooled to word level, and ridge-regression probes are trained on frozen layer-wise representations from Universal Dependencies treebanks. The approach follows structural-probe methodology used to study representational changes during BERT fine-tuning [Merchant et al., 2020](https://arxiv.org/abs/2002.12327).

### 2. Representation similarity analysis

`Rep_Analysis.py` quantifies layer-wise representational drift between Base and fine-tuned models using representational similarity analysis (RSA). The analysis uses XNLI premises and hypotheses, separately stratified by contradiction, neutral, and entailment labels [Conneau et al., 2018](https://aclanthology.org/D18-1269/).

### 3. Logit-lens analysis

`logitlense.py` adapts the multilingual routing-style logit lens of [Schut, Gal, and Farquhar (2025)](https://arxiv.org/abs/2502.15603) to BERT-style masked language models. For POS-controlled English and Chinese cloze prompts, the hidden state at each `[MASK]` position is projected through the model’s MLM head.

The vocabulary is partitioned into English-like and Chinese-like token subsets using Unicode-script heuristics. At every layer, the analysis measures:

- English probability mass, \(p_{en}\).
- Chinese probability mass, \(p_{zh}\).
- Normalised language shares, such as \(p_{zh}/(p_{en}+p_{zh})\).
- Correct-target probability by layer and part of speech.

For two-subtoken Chinese answers, the implementation supports two adjacent mask positions and scores the answer using the geometric mean of the two target-token probabilities.

### 4. Causal tracing

`CausalTracer.py` adapts causal tracing to BERT-style MLM factual retrieval, following the intervention logic used by Schut et al. (2025). We evaluate multilingual factual cloze prompts from mLAMA-style relation datasets, including **Capital**, **Official Language**, **Place of Birth**, **Continent**, and **Developer** [Kassner, Dufter, and Schütze, 2021](https://arxiv.org/abs/2102.00894). Before tracing, facts are filtered so that the Base model ranks the correct masked target among its top-10 predictions. This restricts analysis to facts that the model demonstrably knows. For each fact:

1. Gaussian noise corrupts input embeddings at the subject-token span.
2. The corrupted run establishes a baseline probability for the correct target token.
3. One clean activation is restored at a time at a specific layer and token position.
4. The indirect effect is the recovery in target probability relative to the corrupted baseline.

$$
IE(l,t) = p_{\mathrm{patch}}^{(l,t)}(y) - p_{\mathrm{corr}}(y).
$$

This reveals where clean information can causally restore factual predictions.

### 5. Path patching / attention-head restoration

`PathPatching.py` performs clean-to-corrupted head restoration to identify attention heads that causally carry information from a subject span to a masked target. The method follows the causal intervention principle of path patching and is related to multilingual structural analyses of language models [Zhang et al., 2024](https://arxiv.org/abs/2405.01573).

For each prompt, the code caches clean attention-head outputs, corrupts the subject embeddings with Gaussian noise, and restores one attention head at a time during the corrupted forward pass. A head’s restoration score is the improvement in either correct-target probability or a language-mass statistic:

$$
\Delta_{target}(l,h) = p_{\mathrm{patch}}^{(l,h)}(y) - p_{\mathrm{corr}}(y).
$$

$$
\Delta_{mass}(l,h) = \mathrm{share}_{\mathrm{patch}}^{(l,h)} - \mathrm{share}_{\mathrm{corr}}.
$$

### 6. Steering-vector analysis

The project is motivated in part by the activation-steering result of Schut et al. (2025): in their multilingual LLM experiments, steering vectors computed in English were often more effective than vectors computed in the input/output language. This motivates testing whether bilingual brain-informed fine-tuning changes the directions in activation space associated with language routing or factual retrieval.

**Status:** the current repository contains the logit-lens, causal-tracing, path-patching, structural-probe, and RSA implementations described above. It does not currently include a standalone steering-vector experiment script. Steering is therefore documented here as a planned extension and methodological motivation, rather than reported as a completed result.

## Main findings

Across the implemented analyses, the evidence supports the following interpretation:

- Brain-informed fine-tuning is **largely non-disruptive**: dependency-structure probe performance remains close to the corresponding pretrained Base model.
- Fine-tuning produces **layer- and language-dependent representational drift**: changes are concentrated in later layers for English settings, while Chinese settings show earlier divergence.
- Language-selective fine-tuning can alter **language-specific probability routing**, particularly increasing Chinese-script probability mass for Chinese prompts in the relevant mBERT setting.
- Semantic-selective English tuning can improve **out-of-distribution factual retrieval**, with the clearest gains reported for Capital and Official Language relations.

## Repository structure

```text
.
├── CausalTracer_clean.py       # Activation restoration / causal tracing
├── logitlense_clean.py         # POS-controlled multilingual logit lens
├── PathPatching_clean.py       # Attention-head restoration and mass attribution
├── Probe_analysis_clean.py     # Structural probes and model-weight comparisons
├── Rep_Analysis_clean.py       # RSA and XNLI representation analyses
└── README.md
```

## References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/). NAACL.
- Negi, A., Oota, S. R., Nunez-Elizalde, A. O., Gupta, M., & Deniz, F. (2025). [Brain-Informed Fine-Tuning for Improved Multilingual Understanding in Language Models](https://openreview.net/forum?id=JPogehP8By).
- Merchant, A., Rahimtoroghi, E., Pavlick, E., & Tenney, I. (2020). [What Happens to BERT Embeddings During Fine-tuning?](https://arxiv.org/abs/2002.12327).
- Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S. R., Schwenk, H., & Stoyanov, V. (2018). [XNLI: Evaluating Cross-lingual Sentence Representations](https://aclanthology.org/D18-1269/). EMNLP.
- Kassner, N., Dufter, P., & Schütze, H. (2021). [Multilingual LAMA: Investigating Knowledge in Multilingual Pretrained Language Models](https://arxiv.org/abs/2102.00894).
- Schut, L., Gal, Y., & Farquhar, S. (2025). [Do Multilingual LLMs Think in English?](https://arxiv.org/abs/2502.15603).
- Zhang, R., Yu, Q., Zang, M., Eickhoff, C., & Pavlick, E. (2024). *Structural Similarities and Differences in Multilingual Language Modeling.*
- Nivre, J., et al. (2016). [Universal Dependencies v1: A Multilingual Treebank Collection](https://aclanthology.org/L16-1262/).
