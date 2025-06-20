# eeg-mcp-project

- View the full [sequence diagram here](flow/sequence-diagram.md)

### 🧠 MCP - EEG Data Processing & Local RAG Assistant with Tool Agent
This project integrates:

- EEG Signal Processing (MCP) using BrainFlow.

- RAG (Retrieval-Augmented Generation) for local document-based Q&A using LangChain, Chroma VectorDB, and Ollama (Mistral model).

- Tool-Use Agent — A smart LLM-based controller that decides when to call an EEG API vs directly answering.

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/eeg-mcp-project.git
cd eeg-mcp-project

### 2️⃣ Create & Activate Python Virtual Environment

python -m venv .venv
source .venv/bin/activate  # Mac/Linux

### 3️⃣ Install Python Dependencies
pip install -r requirement.txt

### 4️⃣ Install & Start Ollama (Local LLM)
brew install ollama  # macOS via Homebrew

ollama serve         # Start Ollama server (runs on 127.0.0.1:11434)

ollama pull mistral  # Pull Mistral model for RAG inference

### 5️⃣ Populate Vector Database (One-time Step)
python backend/vectorstore.py

### 6️⃣ Start Servers

## Start MCP EEG Server
python backend/brainflow_server.py

## Start RAG Retrieval Server (LangChain + Ollama)
python backend/rag_server.py

## Start Tool-Agent Controller
python backend/tool_agent_server.py

## Start Node Server
node server.js # mcp-server folder


### ✅ Features Breakdown
1. EEG MCP Server (BrainFlow)

- /read-edf – Read EEG EDF files.

- /visualize-edf – Visualize EEG signals (plot).

- /filter-edf – Apply bandpass filtering to EEG signals.

- /features-edf – Extract average band powers (Delta, Theta, Alpha, Beta, Gamma) from EEG signals.

- /summary-edf – Summarize EEG signal characteristics and metadata

2. RAG Knowledge Assistant (Ollama + Chroma)
- Uses Chroma VectorDB to store documents (docs/brainflow_notes.txt etc.).

- Runs Mistral model via Ollama for local LLM answering.

- API: /mcp/query – Ask questions about EEG concepts, filtering, brainflow usage.

3. Tool-Agent Controller

- A smart agent that:

  Accepts a question and an EDF file

  Decides if it needs to: Call an EEG tool (like /filter-edf, /summary-edf) or just answer using the LLM directly

- Embeds the MCP output in the LLM prompt for more accurate, context-aware responses.

- Endpoint: /mcp/agent - Unified EEG + QA endpoint via intelligent controller

- Example: Ask "Summarize this EEG file" — it calls /summary-edf, gets results, and asks the LLM to explain in human terms.

### 🔄 Restarting Workflow
source .venv/bin/activate

ollama serve

python backend/brainflow_server.py

python backend/rag_server.py

python backend/tool_agent_server.py

node server.js #mcp-server folder

python backend/vectorstore.py #(Optional)

### 🎯 Project Objective
“EEG Data + Local Intelligence” — A private, LLM-powered system for EEG signal interpretation, context-aware document Q&A, filtering, and summaries — all without relying on OpenAI APIs.

### 📝 Notes
- BrainFlow SDK powers EEG data simulation, filtering, feature extraction, and more.

- LangChain + Ollama provide fully local RAG + LLM capabilities.

- Chroma is used to persist and search embedded document vectors.

- This system can route questions dynamically to backend tools and return intelligent LLM-formulated answers.
