# eeg-mcp-project
# 🧠 MCP - EEG Data Processing & Local RAG Retrieval App

This project contains:
**EEG Data Processing MCP (Model Context Protocol) Server** using BrainFlow.
**Local RAG (Retrieval-Augmented Generation)** system that leverages LangChain, Chroma VectorDB, and Ollama for contextual document-based Q&A for EEG.
---

## 🗂️ Project Structure

eeg-mcp-project/
├── backend/
│ ├── brainflow_server.py # MCP APIs (read, visualize, filter, features)
│ ├── rag_server.py # RAG API server using Ollama LLM + Vector DB
│ ├── vectorstore.py # Vector DB Population script
│ └── .env # API keys & env variables (ignored in git)
├── uploads/ # Temp uploads folder
├── data/ # EDF EEG data files
├── vectorstore/ # Persisted Chroma Vector DB files
├── requirements.txt # Python dependencies list
├── .gitignore # Ignore node_modules, .env, cache files
└── README.md # This file

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/eeg-mcp-project.git
cd eeg-mcp-project

### 2️⃣ Setup Python Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

### 3️⃣ Install Python Dependencies
pip install -r requirement.txt

### 4️⃣ Install & Start Ollama (Local LLM)
brew install ollama  # macOS via Homebrew
ollama serve         # Start Ollama server (runs on 127.0.0.1:11434)
ollama pull mistral  # Pull Mistral model for RAG inference

### 5️⃣ Populate Vector Database (One-time Step)
python backend/vectorstore.py

### 6️⃣ Start MCP Backend Server
python backend/brainflow_server.py

### 7️⃣ Start RAG Server (LangChain + Ollama)
python backend/rag_server.py

### 8️⃣ (Optional) Node.js Forwarding Server
cd frontend/
npm install
npm start

### ✅ Features Breakdown
1. EEG MCP Server (BrainFlow)

/read-edf – Read EEG EDF files.

/visualize-edf – Visualize EEG signals (plot).

/filter-edf – Apply bandpass filtering to EEG signals.

/features-edf – (Optional) Extract band power features.

2. RAG Knowledge Assistant (Ollama + Chroma)
Uses Chroma VectorDB to store documents (docs/brainflow_notes.txt etc.).

Runs Mistral model via Ollama for local LLM answering.

API: /mcp/query – Ask questions about EEG concepts, filtering, brainflow usage.

3. Unified API Gateway (server.js)
MCP endpoints (read, visualize, filter)

RAG endpoint (/mcp/query) for knowledge QA.

Acts as a single entry point but routes to independent backends.


### 🔄 Restarting Workflow
Whenever you return to work:
# Activate environment
source .venv/bin/activate

# Start Ollama server
ollama serve

# Start MCP server
python backend/brainflow_server.py

# Start RAG server
python backend/rag_server.py

# (Optional) Populate VectorDB again if updated
python backend/vectorstore.py

# (Optional) Run Node.js server
cd frontend/
npm start

### 🎯 Goal of This Project
"EEG Signal Processing (MCP) + Local Doc-based Assistant (RAG) for EEG"

### ✅ Important Notes

MCP uses BrainFlow SDK for EEG Data acquisition & filtering.

RAG system uses Ollama (local LLM inference) with VectorDB (Chroma).

No OpenAI API keys needed (runs locally via Ollama).