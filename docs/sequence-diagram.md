# ToolAgent Sequence Diagram

```mermaid
sequenceDiagram
    participant Client
    participant ToolAgent
    participant MCP_Server
    participant LLM (Ollama)

    Client-->>ToolAgent: POST /mcp/agent (question + file)

    alt Tool needed (e.g., "filter", "band", "summary")
        ToolAgent-->>MCP_Server: Call appropriate MCP API (e.g., /filter-edf)
        MCP_Server-->>ToolAgent: Returns JSON data
        ToolAgent-->>LLM (Ollama): Ask question + MCP data in prompt
        LLM (Ollama)-->>ToolAgent: Final answer
    else General question (no file needed)
        ToolAgent-->>LLM (Ollama): Ask question directly
        LLM (Ollama)-->>ToolAgent: Final answer
    end

    ToolAgent-->>Client: Respond with answer
