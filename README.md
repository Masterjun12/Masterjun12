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
```

Research questions:

- Can KG embeddings preserve global graph structure better than retrieval-only GraphRAG?
- How many graph tokens are useful for LLM reasoning?
- Should graph tokens represent global triples, head-relation views, or relation-tail views?
- Can frozen LLMs use projected graph representations without full fine-tuning?
- How should KG-LLM systems be evaluated on multi-hop QA?

---

### 2. GraphRAG, Legal QA, and Multi-Hop Evaluation

I have worked on building graph-based retrieval systems using legal documents and structured triples.

Main interests:

- Legal document parsing
- Entity and relation extraction
- SPO triple construction
- Neo4j-based graph storage
- OpenSearch-based vector retrieval
- Graph + vector hybrid retrieval
- Multi-hop QA dataset construction
- Retrieval result evaluation

System direction:

```text
Legal Documents
    ↓
Chunking / Parsing
    ↓
Entity-Relation Extraction
    ↓
Knowledge Graph Construction
    ↓
Neo4j + OpenSearch
    ↓
Graph Retrieval + Vector Retrieval
    ↓
LLM-based Legal QA
```

I focus on questions that require crossing document boundaries, such as:

- 0-hop: direct factual lookup
- 1-hop: entity-to-law or entity-to-case relation
- 2-hop: case → issue → law / precedent → legal concept

---

### 3. Small Language Model Optimization

I am interested in making LLM-based systems practical under limited resources.

Main topics:

- Small Language Models
- Parameter-Efficient Fine-Tuning
- LoRA / PEFT
- Memory-efficient inference
- Quantization-aware deployment
- vLLM-based serving
- RAG optimization for compact models
- Lightweight agentic workflows

Key motivation:

> In many enterprise and closed-network environments, cost, latency, deployment constraints, and maintainability are often more important than simply using the largest model.

---

### 4. LLM Agents and Agentic Workflow Systems

I explore LLM agents as **stateful, tool-using, workflow-controlled systems**, not just prompt chains.

Main topics:

- LangGraph-based agent orchestration
- Agent-to-Agent workflow design
- MCP-style tool integration
- A2A-style agent interoperability
- Human-in-the-loop control
- Agent memory and state management
- Tool calling and structured output
- Enterprise workflow automation

Example system pattern:

```text
User Request
    ↓
Planner Agent
    ↓
Task-Specific Agents
    ├── Search Agent
    ├── DB Agent
    ├── Report Agent
    ├── Evaluation Agent
    └── Deployment Agent
    ↓
State Store / Version DB
    ↓
Final Output + Logs + Feedback Loop
```

I am especially interested in building agent systems that can operate in enterprise environments:

- internal document search
- graph/database querying
- report generation
- monitoring and notification
- experiment automation
- reproducible AI pipelines

---

### 5. Reinforcement Learning for LLM Post-Training

I have explored GRPO-style post-training experiments for mathematical reasoning tasks.

Main topics:

- GRPO
- reward design
- group-level optimization
- adaptive thresholding
- group composition-aware learning
- evaluation harness construction
- reward / loss / generation logging

Current research idea:

> Groups with mixed correct and incorrect samples may contain more useful learning signal than groups where all samples are correct or all samples are wrong.

Example direction:

```text
Prompt
    ↓
Generate Group of Responses
    ↓
Reward Evaluation
    ↓
Analyze Group Composition
    ↓
Adjust Update Strength
    ↓
Policy Optimization
```

Evaluation interests:

- accuracy
- sample usage
- reward distribution
- group correctness ratio
- token efficiency
- learning stability
- comparison with vanilla GRPO

---

### 6. Computer Vision & Medical AI

I have worked on lightweight medical image segmentation, especially brain tumor segmentation.

Main topics:

- U-Net variants
- Depthwise convolution
- Lightweight segmentation
- Medical image analysis
- MRI brain tumor segmentation
- Resource-constrained clinical AI

Representative work:

- **LAG-UNet: Optimized Brain Tumor Segmentation for Resource-Constrained Clinical Environments via Adaptive Gating**

Key idea:

> Reduce model size and memory usage while preserving or improving segmentation quality for clinical environments with limited computational resources.

---

### 7. Generative AI and Content Automation

I am also interested in automated content generation pipelines using AI agents.

Main topics:

- scenario generation
- character database
- storyboard generation
- text-to-image prompting
- text-to-video prompting
- voice generation
- post-processing
- publishing automation
- content monitoring

Example architecture:

```text
Trend Collector
    ↓
Scenario Agent
    ↓
Character / Relationship DB
    ↓
Storyboard Agent
    ↓
Image / Video / Voice Generation
    ↓
Post-processing Agent
    ↓
Publishing Agent
    ↓
Analytics Feedback
```

This direction connects generative AI with agentic automation and production pipelines.

---

## 🧩 Current Projects

### KG-LLM Token Injection

A research direction for injecting knowledge graph representations into LLMs.

Core idea:

- Train or load KG embeddings
- Project graph vectors into LLM embedding space
- Concatenate graph tokens with question tokens
- Keep the LLM frozen or LoRA-adapted
- Compare against GraphRAG and prompt-only baselines

---

### Legal GraphRAG System

A graph-based legal QA system using structured legal data.

Components:

- legal document parser
- SPO triple extraction
- Neo4j graph database
- OpenSearch vector index
- hybrid graph/vector retriever
- multi-hop legal QA benchmark
- evaluation harness

---

### GRPO Improvement Experiments

A post-training experiment pipeline for LLM reasoning.

Focus:

- vanilla GRPO baseline
- adaptive threshold GRPO
- group composition-aware update weighting
- detailed logging of rewards, losses, generations, and correctness
- reproducible experiment reports and tables

---

### Enterprise AI Agent MVP

A LangGraph-based AI agent system for public information search and institutional guidance.

Features:

- notice search
- academic schedule guidance
- public service information
- department/contact routing
- React-style UI
- OpenWebUI-like interaction design
- role-based mock login
- LangGraph workflow backend

---

### LAG-UNet Brain Tumor Segmentation

A lightweight segmentation model for brain tumor MRI analysis.

Focus:

- parameter reduction
- memory efficiency
- adaptive gating
- comparison with vanilla U-Net
- clinical resource-constrained deployment

---

## 📚 Paper Reading and Research

### Graph & LLM / Reasoning

- [Knowledge Graph Large Language Model (KG-LLM) for Link Prediction](https://arxiv.org/abs/2403.07311)
- [Knowledge Graphs as Context Sources for LLM-Based Explanations of Learning Recommendations](https://ieeexplore.ieee.org/abstract/document/10578654)
- [Generate-on-Graph: Treat LLM as both Agent and KG for Incomplete Knowledge Graph Question Answering](https://aclanthology.org/2024.emnlp-main.1023/)
- [Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph](https://arxiv.org/abs/2307.07697)

### GraphRAG / Agentic Retrieval

- Microsoft GraphRAG
- HippoRAG
- GNN-RAG
- Think-on-Graph
- Tool-augmented graph reasoning
- Hybrid vector and graph retrieval

### LLM Agents and System Design

- LangGraph
- Model Context Protocol
- Agent-to-Agent communication
- Tool calling
- Long-running stateful agents
- Agent evaluation and observability

### LLM Optimization and Post-Training

- LoRA / PEFT
- RLHF / RLAIF
- PPO
- GRPO
- Reward design
- Efficient inference
- vLLM serving

### Computer Vision & Medical AI

- [A Survey of Methods for Brain Tumor Segmentation Based on MRI Images](https://github.com/Masterjun12/Paper-and-experiment-seminar/blob/main/Paper/A%20survey%20of%20methods%20for%20brain%20tumor%20segmentation-based%20MRI%20images%20.pdf)
- [Edge U-Net: Brain Tumor Segmentation Using MRI Based on Deep U-Net Model with Boundary Information](https://github.com/Masterjun12/Paper-and-experiment-seminar/blob/main/Paper/Edge_U-Net_Brain_tumor_segmentation_using_MRI_based_on_Deep_U-Netmodel_with_boundary_information__.pdf)
- [FU-net: Multi-class Image Segmentation Using Feedback Weighted U-net](https://github.com/Masterjun12/Paper-and-experiment-seminar/blob/main/Paper/FU-net_Multi-class_Image_Segmentation_Using_Feedback_Weighted_U-net.pdf)
- [Using a Generative Adversarial Network to Generate Synthetic MRI Images for Multi-class Automatic Segmentation of Brain Tumors](https://github.com/Masterjun12/Paper-and-experiment-seminar/blob/main/Paper/Using_a_generative_adversarialnetwork_to_generate_synthetic_MRIimages_for_multi-class_automaticsegmentation_of_brain_tumors.pdf)
- [Edge-Boosted U-Net for 2D Medical Image Segmentation](https://github.com/Masterjun12/Paper-and-experiment-seminar/blob/main/Paper/_Edge-Boosted_U-Net_for_2D_Medical_Image_Segmentation.pdf)
- [A Survey on Efficient Vision Transformers: Algorithms, Techniques, and Performance Benchmarking](https://arxiv.org/abs/2309.02031)

More papers:

- [Paper and Experiment Seminar Repository](https://github.com/Masterjun12/Paper-and-experiment-seminar/tree/main/Paper)

---

## 🧰 Languages and Libraries

### Programming

- Python
- Bash
- SQL

### Deep Learning

- PyTorch
- TensorFlow
- Keras
- Hugging Face Transformers
- PEFT / LoRA

### Data Science

- Pandas
- NumPy
- Scikit-learn

### NLP / LLM

- LangChain
- LangGraph
- LlamaIndex
- vLLM
- OpenWebUI
- RAG
- GraphRAG
- Tool Calling
- Structured Output Generation

### Graph and Databases

- Neo4j
- OpenSearch
- SQL / SQLGate
- Knowledge Graph Embedding
- Vector Search
- Hybrid Retrieval

### Computer Vision

- OpenCV
- Ultralytics
- U-Net
- DeepLab
- Mask R-CNN
- Medical Image Segmentation

### Infrastructure / MLOps

- Docker
- Docker Compose
- Kubernetes
- Linux Server Environment
- Nginx
- Jenkins
- Nexus
- REST API
- WebSocket / API Integration

---

## ⚙️ System Design Interests

I am interested in building AI systems as complete pipelines:

```text
Data Source
    ↓
Parser / Ingestion
    ↓
Database / Vector Store / Graph DB
    ↓
Retriever / Agent Tool
    ↓
LLM Serving
    ↓
Workflow Orchestration
    ↓
Frontend UI
    ↓
Monitoring / Evaluation / Feedback
```

This includes:

- closed-network deployment
- Docker-based reproducibility
- model serving with vLLM
- OpenAI-compatible local endpoints
- graph/vector DB integration
- agent workflow debugging
- frontend monitoring interfaces
- experiment automation

---

## 🎯 Research Direction Summary

My long-term research direction is to build **knowledge-grounded, efficient, and agentic AI systems**.

In particular, I am interested in the intersection of:

```text
Knowledge Graphs
    ×
Small / Efficient Language Models
    ×
Agentic Workflows
    ×
Enterprise AI Infrastructure
    ×
Reliable Evaluation
```

The goal is to move beyond simple prompt engineering and build AI systems that are:

- structured
- explainable
- reproducible
- resource-efficient
- deployable
- connected to real data and tools

---

## 📫 Contact

- LinkedIn: [Yurim Oh](https://www.linkedin.com/in/yurim-oh-225001285/)
- Email: ak333ak12@jj.ac.kr

---

<p>
  <img align="center" src="https://github-readme-stats.vercel.app/api?username=masterjun12&show_icons=true" alt="masterjun12" />
</p>
