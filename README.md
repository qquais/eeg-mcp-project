# eeg-mcp-project

* View the full [sequence diagram here](flow/sequence-diagram.md)

### 🧠 EEG MCP Server & Intelligent Tool (EEG-BOT)

This project integrates:

* EEG Signal Processing (MCP) using **BrainFlow** and **MNE-Python**.

* RAG (Retrieval-Augmented Generation) for local document-based Q\&A using LangChain, FAISS VectorDB, and Ollama (nomic-embed-text for embeddings, Mistral for answers).

* Tool-Use Agent — A smart LLM-based controller that decides when to call an EEG API, retrieves its output, and intelligently answers using both the tool response and contextual EEG knowledge.

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/eeg-mcp-project.git
cd eeg-mcp-project
```

### 2️⃣ Create & Activate Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Install & Start Ollama (Local LLM)

```bash
brew install ollama         # macOS via Homebrew
ollama serve                # Start Ollama server (127.0.0.1:11434)
ollama pull mistral         # Pull Mistral model
ollama pull nomic-embed-text  # Pull embedding model
```

### 5️⃣ Populate Vector Database (One-time Step)

```bash
python backend/vectorstore.py
```

### 6️⃣ Start Servers

```bash
# Start EEG MCP Server (BrainFlow + MNE)
python backend/mcp_server.py

# Start Tool-Agent Controller
python backend/tool_agent_server.py

# Start NodeJS Server (Optional, after UI Implementation)
cd frontend
node server.js
```

### ✅ Features Breakdown

1. **EEG MCP Server (BrainFlow & MNE)**

* `/read-edf` – Read EEG EDF files and metadata
* `/visualize-edf` – Plot EEG signal previews
* `/psd-edf` – Compute & plot Power Spectral Density (Band Powers)
* `/filter-edf` – Apply bandpass filter (0.5 Hz - 40 Hz)
* `/features-edf` – Get brainwave band powers (Delta to Gamma)
* `/summary-edf` – Per-channel signal summary (mean, std, min, max)

2. **RAG Assistant (LangChain + FAISS + Ollama)**

* Stores `./docs/*.txt` files in FAISS vector DB
* Embeds using `nomic-embed-text` (via Ollama)
* Generates answers with `mistral` model (via Ollama)
* Endpoint: `retrieve_context(question)` used by Tool Agent

3. **Tool-Agent Controller (`/mcp/agent`)**

* Accepts: question + EEG file
* Detects which EEG API to call based on keywords
* Merges EEG output + RAG context into LLM prompt
* Returns interpreted human-readable answer
* Visualization questions generate image + download link

4. **Optional UI Integration (Node Server)**

* The `frontend/server.js` Node server bridges EEG APIs with frontend apps (e.g., React)
* Enables upload previews, waveform rendering, or future real-time dashboard integration
* Not required for backend+LLM functionality but useful for full-stack development

### 🔄 Restart Workflow

```bash
python -m venv .venv
source .venv/bin/activate
ollama serve
python backend/mcp_server.py
python backend/tool_agent_server.py
cd frontend && node server.js #Optional
```

### Example Prompts

* "Summarize this EEG signal"
* "Show the PSD of this file"
* "What is the alpha band power?"
* "Filter this EEG signal"
* "Plot the waveform"

###  Project Objective

“EEG Data + Local Intelligence” — A fully local EEG analysis platform with:

* Smart LLM-assisted interpretation
* Modular APIs for future ML/real-time integration
* No OpenAI/API keys needed

###  Notes

* EEG: BrainFlow for data emulation, MNE for file parsing
* LLM: Ollama for local `mistral` model
* RAG: FAISS for local doc search using `nomic-embed-text`
* Tool Agent: Connects LLM to EEG tools
* Visualization is served back as downloadable image

###  Project Demo 

* Link: 