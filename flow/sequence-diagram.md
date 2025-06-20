# ToolAgent Sequence Diagram

```mermaid
sequenceDiagram
    participant Client
    participant Tool Agent
    participant MCP Server
    participant LLM (Ollama)

    Client->>Tool Agent: POST /mcp/agent (question + file)

    alt EEG Tool Required (e.g., filter, band, summary)
        Tool Agent->>MCP Server: Call relevant EEG API (e.g., /filter-edf)
        MCP Server-->>Tool Agent: Return JSON/Image Data
        Tool Agent->>LLM (Ollama): Ask with EEG data + context
        LLM (Ollama)-->>Tool Agent: Final response
    else General Question
        Tool Agent->>LLM (Ollama): Ask using context only
        LLM (Ollama)-->>Tool Agent: Final response
    end

    Tool Agent-->>Client: Return answer (and image URL if applicable)

