# 🎥 Multilingual YouTube RAG Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** web application built with **Python, LangChain, FAISS, Streamlit, and Google Gemini** that lets users query YouTube videos in **English**, **Hinglish**, and **Hindi (Devanagari)** with grounded, streaming responses.

🔗 **Live Demo:** [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app) *(Replace with your deployed Streamlit Cloud URL)*

---

## ✨ Key Features

* **Multilingual & Script-Aware Q&A:** Automatically detects and mirrors the language and script of the user's prompt—answering in **English**, **Hinglish (Roman script)**, or **Devanagari Hindi** in smooth, cohesive paragraphs even when the source video transcript is in English.
* **Resilient Transcript Ingestion (`youtube_utils.py`):** Extracts 11-character video IDs across standard `watch`, `youtu.be`, `shorts`, `embed`, and `live` URLs, with automatic fallback to translatable caption tracks when a preferred language is unavailable.
* **Semantic Chunking & FAISS Vector Search:** Splits transcripts into overlapping context-preserving chunks (`RecursiveCharacterTextSplitter`) and indexes dense embeddings (`gemini-embedding-2`) inside a cached **FAISS** similarity store (`@st.cache_resource`).
* **Production Reliability (Retries + Model Fallback):** Handles transient `503` high-demand spikes and `429` rate limits using exponential backoff retries and automatic model fallback (`gemini-3.5-flash-lite` → `gemini-3.6-flash` / `gemini-3.8-flash`).
* **Cloud-Safe Quick Demo Mode:** Bundles pre-cached sample video transcripts with one-click starter questions so recruiters and visitors can test the full RAG workflow immediately—even if YouTube rate-limits datacenter IPs.
* **Prompt-Injection & Input Guardrails:** Separates untrusted transcript/user content (`<transcript_data>` and `<user_question>` boundaries) from system instructions and enforces input sanitization.

---

## 🏗️ System Architecture

```text
YouTube URL / ID  ──►  extract_video_id() & get_transcript()  ──►  LangChain Document
  (or Quick Demo)          (Language fallback + metadata)                 │
                                                                          ▼
                                                        RecursiveCharacterTextSplitter
                                                          (chunk_size=1000, overlap=200)
                                                                          │
                                                                          ▼
                                                       Gemini Embeddings (gemini-embedding-2)
                                                                          │
                                                                          ▼
User Question  ──►  FAISS Similarity Retriever (Top-k=4)  ──►  Context-Grounded Prompt
(EN / Hinglish / HI)                                                      │
                                                                          ▼
                                                    Gemini LLM (Retry + Fallback + Streaming)
                                                                          │
                                                                          ▼
                                                    Grounded Answer + Source Context Expander