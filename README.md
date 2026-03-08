# 🔬 Binding Circuit Discovery Pipeline (BCDP)

A mechanistic interpretability pipeline for discovering **binding circuits in transformer language models**.

Binding, the ability to associate entities with roles, attributes, or relations, is a fundamental cognitive operation required for structured reasoning and language understanding.

Recent interpretability work suggests that binding information is represented in **low-dimensional subspaces of the residual stream**, but we still lack a systematic way to identify **which components create, update, and use these representations**.

The **Binding Circuit Discovery Pipeline (BCDP)** is a unified pipeline designed to discover and validate the **circuits responsible for binding inside transformer models**.

---

# 🔁 Pipeline Overview

```
                ┌──────────────────────────────┐
                │   Contrastive Binding Data   │
                │ (minimal binding datasets)   │
                └───────────────┬──────────────┘
                                │
                                ▼
               ┌────────────────────────────────┐
               │  Activation Tracing (Residual) │
               │  Extract activations per layer │
               └───────────────┬────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │        Subspace Discovery            │
            │  diff-means / DBCM / PCA / DAS       │
            │  → binding-relevant residual space   │
            └───────────────┬──────────────────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼

   ┌─────────────────────┐      ┌────────────────────────┐
   │  Attention Analysis │      │   MLP Writer Analysis  │
   │                     │      │                        │
   │ Rank heads by how   │      │  Project MLP weights   │
   │ much they route     │      │  onto binding subspace │
   │ information into S  │      │                        │
   └───────────┬─────────┘      └──────────────┬─────────┘
               │                               │
               └──────────────┬────────────────┘
                              ▼

                ┌──────────────────────────────────┐
                │      Binding Circuit Model       │
                │                                  │
                │   Subspace  ←→  Heads  ←→  MLPs  |
                │                                  │
                └──────────────┬───────────────────┘
                               │
                               ▼

                ┌─────────────────────────────┐
                │     Causal Validation       │
                │                             │
                │  • Subspace ablation        │
                │  • Head ablation            │
                │  • Writer ablation          │
                │  • Sufficiency tests        │
                └─────────────────────────────┘
```

---

# 🧠 Core Idea

BCDP reconstructs binding mechanisms by identifying three interacting components:

### Residual Stream Subspaces
Low-dimensional directions encoding binding information.

### Attention Heads
Components that **route binding information between tokens**.

### MLP Writers
Low-rank operators that **write binding updates into the residual stream**.

Together, these elements form a **binding circuit spanning multiple layers of the transformer**.

---

# ⚙️ Pipeline Stages

### 1️⃣ Binding Subspace Discovery

Identify candidate binding subspaces in the residual stream using methods such as:

- Difference-in-Means
- Desiderata-Based Component Masking (DBCM)
- PCA / SVD
- Distributed Alignment Search (DAS)

---

### 2️⃣ Attention Head Ranking

Measure how strongly each attention head **routes information into the binding subspace**.

Heads are ranked according to their projected contribution to the discovered subspace.

---

### 3️⃣ MLP Writer Identification

Detect MLP components that write binding information by:

- projecting weight matrices onto the binding subspace
- applying **SVD** to detect low-rank operators

---

### 4️⃣ Circuit Reconstruction

Combine subspaces, attention heads, and MLP writers into a unified **binding circuit** across transformer layers.

---

# 🧪 Experiments

The pipeline supports causal interpretability experiments such as:

- Subspace **necessity tests**
- Subspace **sufficiency tests**
- **Attention head ablations**
- **MLP writer ablations**
- Circuit reconstruction validation

These experiments test whether identified components are **mechanistically required** for binding behavior.

---

# 🎯 Research Question

BCDP aims to serve as a proof of concept that leveraging geometric insights can enhance and improve circuit discovery. 

Our main research question is: "Can the geometric structure of binding representations be systematically leveraged to identify and causally validate the circuits that implement binding in transformer language models?"

---

# 📂 Repository Structure

```
BCDP
│
├── data/           # Binding datasets
├── trace/          # Activation tracing
├── subspace/       # Subspace discovery methods
├── intervention/   # Causal interventions
├── ranking/        # Head and MLP ranking
├── experiments/    # Main experiments
├── models/         # Model interfaces
└── utils/          # Utilities
```

---

# 🚧 Project Status

Work in progress.

Current focus:

- scalable subspace discovery
- circuit reconstruction
- causal validation experiments

---

# 👤 Author

**Omer Naziri**  
Tel Aviv University
