### Yurim Oh [@Masterjun12]

<p align="left">
  <img src="https://komarev.com/ghpvc/?username=masterjun12" alt="masterjun12" />
</p>

---

## 🧠 AI Researcher & System Builder

I am interested in building **practical AI systems that connect models, knowledge, tools, and infrastructure**.

My current focus lies in:

- **Graph-based AI and Knowledge-Augmented LLMs**
- **LLM Agents, Agentic Workflows, and Enterprise AI Systems**
- **Small Language Model Optimization and Efficient Inference**
- **Reinforcement Learning for LLM Post-Training**
- **Medical AI and Lightweight Vision Models**

Rather than treating LLMs as isolated chat models, I focus on how they can be connected with
**knowledge graphs, retrieval systems, databases, agents, serving infrastructure, and evaluation pipelines**.

---

## 📜 Qualifications

- NVIDIA — Fundamentals of Deep Learning
- NVIDIA — Transformer Based Natural Language Processing Models
- Microsoft Certified — Azure AI Fundamentals
- Microsoft Certified — Azure Data Fundamentals

---

## 🔭 Career

- B.S. in Artificial Intelligence, Jeonju University, Korea, 2025
- M.S. student in Agro AI, Jeonju University, Korea, 2024 ~ Present
- Completed an education certificate program at the University of Toronto C-MORE Lab
- Research Intern, Dareesoft through the WEMEET program
- Research Intern, Rural Development Administration through the WEMEET program

---

## 🌱 Research Themes

### 1. Graph-based AI & Knowledge-Integrated LLMs

I am especially interested in connecting **structured knowledge** with language models.

Main topics:

- Knowledge Graph Embedding: `TransE`, `RotatE`, `DistMult`, `ComplEx`
- Graph Neural Networks and Graph Transformers
- GraphRAG and graph-based retrieval
- Multi-hop reasoning over structured knowledge
- Legal knowledge graph construction and legal QA
- KG-to-LLM representation injection

Current direction:

> Instead of only retrieving graph fragments as text, I explore how **KG embeddings can be projected into the LLM embedding space** and used as additional graph tokens for reasoning.

Example architecture:

```text
Knowledge Graph
    ↓
KGE Model: TransE / RotatE / DistMult
    ↓
Graph Vector s
    ↓
Projection Layer: g = W s + b
    ↓
[Graph Tokens] + [Question Tokens]
    ↓
Frozen LLM / LoRA-adapted LLM
    ↓
Answer Generation