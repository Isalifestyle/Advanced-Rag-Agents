# 🧠 Advanced RAG Agents — Legal and Scientific Assistants

This project implements two intelligent agents using **advanced Retrieval-Augmented Generation (RAG)** techniques:

1. A **Legal Assistant Agent**
2. An **AI-Driven Scientific Research Assistant**

Built with a focus on `Query Rewriting`, `HyDE`, `LangGraph`, and `LangSmith` tracing, this project showcases agentic reasoning applied to real-world legal and scientific domains.

---

## 🧾 Project Overview

This project was developed as part of **Mini-Project 2: Building Agents using Advanced RAG Techniques**, where the goal was to:

* Simulate advanced information retrieval pipelines
* Improve performance through query rewriting, HyDE, speculative generation, and human-in-the-loop mechanisms

### 🔧 Technologies Used

* Python
* LangGraph / LangChain
* Tavily Web Scraper
* LangSmith (for tracing)
* Streamlit (for UI testing)
* FAISS (for vector storage)

---

## 🧑‍⚖️ Legal Assistant Agent (Problem 1)

### 🧩 Pipeline Design

Implements the **HYDE technique**:

1. Generate a hypothetical answer based on the query
2. Insert that response into the vector store
3. Retrieve context-specific documents using the hypothetical response

> The vector store includes documents specifically focused on **real estate law in North Carolina**.

### 🧪 Test Case Summary

**Query:** Do I need to be a real estate agent to buy a home in NC?

**Summary Output:**

* Documents referenced pertain to **timeshare resale law** (G.S. 93A-65)
* No document explicitly states that a real estate agent is needed to buy a home
* Model properly adds a **disclaimer** advising consultation with a legal professional

### 🚨 Known Issues

* **Streamlit's human input** functionality caused the search to restart unexpectedly
* Without Streamlit, the human-in-the-loop logic works as intended

---

## 🧪 AI Research Assistant (Problem 2)

### 🧩 Pipeline Design

* One main graph with both **query expansion** and **RAG response generation**
* Incorporates **Tavily Web Scraper** for broader results on emerging topics

### 🔬 Test Case Summary

**Query:** How do LLMs process and generate text?

**Summary Output Highlights:**

* Describes **pre-training**, **fine-tuning**, **masked modeling**, and **unified objectives**
* Explains **context handling**, **scaling laws**, and **limitations** (e.g. hallucinations, context window)
* Clearly outlines **dataset types**, **RLHF**, and **bias issues**

### 🚨 Known Issues

* High latency during long runs due to **vector database creation**
* LangGraph subgraph features were explored but not used due to complexity

---

## 🧪 Observations on Performance

* The **HYDE-based Legal Agent** tends to produce more robust results compared to simple query rewriting
* LangSmith tracing was **critical** in diagnosing flow, bottlenecks, and improving prompt structure
* Pipeline runs were effective, but **model generation latency remains an issue** in both agents

---

## 🗂 Submission Summary

* ✅ `.ipynb` and `.py` implementations included
* ✅ Streamlit-based interface for testing
* ✅ Vector store indexing and LangSmith tracing integrated
* ✅ Complete write-up and test case summaries

---

## 🧪 Grading Rubric Coverage

* ✅ Agentic RAG techniques (HYDE, query rewrite, speculative)
* ✅ Clear explanation and evaluation
* ✅ Test case results and performance discussion

---

## 🧑‍💻 Author Notes

This project involved iterative debugging, experimentation with LangGraph, and adapting to real-world performance limits while implementing advanced RAG structures.
