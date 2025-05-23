
# AI Occupational Capability Index (AI-OCI)

**A Novel Framework for Assessing AI's Task-Level Impact on Human Labor**  
Conference: [AIES 2025 – AAAI/ACM Conference on AI, Ethics, and Society](https://www.aies-conference.com/2025/call-for-papers/)

## Overview

The **AI Occupational Capability Index (AI-OCI)** is a dynamic, task-level framework for quantifying the alignment between artificial intelligence capabilities and human occupational tasks. Unlike prior methods that rely on expert heuristics or occupation-level classification, AI-OCI embeds both AI capabilities and O*NET occupational tasks using state-of-the-art language models to measure their semantic alignment. This enables interpretable, scalable assessments of AI’s potential to augment or substitute human labor.

This repository supports the analysis and visualizations presented in our AIES 2025 paper submission.

---

## Key Features

- 🔍 **Task-Level Alignment**: Embeds and compares over 19,000 O*NET tasks with 338 AI capabilities.
- 🤖 **Multi-Model Embeddings**: Includes support for GPT-based embeddings, BERT, and E5 multilingual models.
- 📊 **Clustering and Dimensionality Reduction**: Applies PCA and KMeans to analyze and visualize AI-human task similarities.
- 📈 **Labor Market Impact Analysis**: Correlates AI-OCI with wage and employment data across Pre-COVID, COVID, and LLM-adoption eras.
- ✅ **Validated Against Benchmarks**: Empirically compared with AIOE, GPT-4 Beta, and Frey’s automation index.

---

## Repository Structure

```
AI-OCI/
├── process_data_pca_gpt_8_updated_granular_main.py   # Main preprocessing and embedding script
├── process_cluster_analysis_granular.py              # Clustering, plotting, and regression analysis
├── data/                                             # Contains input data (O*NET, capability sets, embeddings, etc.)
├── imgs/                                             # Output figures used in the paper
├── utils/                                            # Helper functions and plotting modules
├── requirements.txt                                  # Required Python packages
└── README.md                                         # This file
```

---

## Setup and Dependencies

This project uses Python 3.8+. Install all dependencies using:

```bash
pip install -r requirements.txt
```

Key libraries include:
- `pandas`, `numpy`, `scikit-learn`
- `matplotlib`, `seaborn`
- `openai`, `sentence-transformers`
- `tqdm`, `umap-learn`

Make sure you have access to API keys or pre-downloaded embeddings if not generating from OpenAI models directly.

---

## Reproducibility Instructions

### Step 1: Generate AI-OCI Scores

Run the main processing script to:
- Load AI capability and O*NET task data
- Generate embeddings
- Apply PCA for dimensionality reduction
- Compute cosine similarities
- Export granular AI-OCI scores

```bash
python process_data_pca_gpt_8_updated_granular_main.py
```

Outputs:
- `ai_oci_scores_granular.csv`
- `pca_components.pkl`
- `embeddings_dict.pkl`

### Step 2: Analyze and Visualize

Use the second script to:
- Cluster embeddings
- Correlate AI-OCI with labor market trends
- Produce visuals for task similarity, wage/employment trends, and benchmark comparisons

```bash
python process_cluster_analysis_granular.py
```

Outputs:
- PNG/PDF plots saved to `imgs/`
- Clustering summaries
- Correlation matrices with AIOE, Frey Index, GPT-4 Beta

---

## Citation

If you use this framework or codebase in your research, please cite:

```
@inproceedings{awuahgyasi2025aioci,
  title={AI-OCI: A Task-Level Framework for Assessing AI’s Occupational Capabilities},
  author={Awuah-Gyasi, Freddie and [Co-authors]},
  booktitle={AAAI/ACM Conference on AI, Ethics, and Society (AIES)},
  year={2025}
}
```

---


