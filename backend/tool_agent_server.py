from flask import Flask, request, jsonify
import requests
from langchain_community.llms import Ollama
from vectorstore import retrieve_context
import numpy as np
import ast

app = Flask(__name__)
llm = Ollama(model="mistral")

# Tool Mapping
TOOL_MAPPING = {
    "band": "/features-edf",
    "feature": "/features-edf",
    "brainwave": "/features-edf",
    "power": "/features-edf",
    "filter": "/filter-edf",
    "noise": "/filter-edf",
    "visualize": "/visualize-edf",
    "plot": "/visualize-edf",
    "summary": "/summary-edf",
    "describe": "/summary-edf",
    "overview": "/summary-edf",
    "psd": "/psd-edf",
    "spectrum": "/psd-edf",
    "spectral": "/psd-edf",
    "frequency": "/psd-edf"
}

def pick_tool(question):
    for keyword, endpoint in TOOL_MAPPING.items():
        if keyword in question.lower():
            return endpoint
    return None

def summarize_filtered_data(filtered_data):
    summary = {}
    for channel, values in filtered_data.items():
        arr = np.array(values)
        summary[channel] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
    return summary

@app.route("/mcp/agent", methods=["POST"])
def agent():
    file = request.files.get("file")
    question = request.form.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    tool_endpoint = pick_tool(question)
    rag_context = retrieve_context(question)

    if tool_endpoint and file:
        files = {"file": (file.filename, file.stream, file.mimetype)}
        mcp_res = requests.post(f"http://localhost:5000{tool_endpoint}", files=files)

        if mcp_res.status_code != 200:
            return jsonify({"error": f"Tool call failed at {tool_endpoint}"}), 500

        content_type = mcp_res.headers.get("Content-Type", "")

        # PSD: image + band powers in header
        if tool_endpoint == "/psd-edf" and "image" in content_type:
            band_powers_header = mcp_res.headers.get("X-Band-Powers")
            if band_powers_header:
                try:
                    band_powers = ast.literal_eval(band_powers_header)
                except Exception:
                    band_powers = band_powers_header
                final_prompt = f"""
Context:
{rag_context}

User Question:
{question}

MCP Band Powers:
{band_powers}

Interpret this EEG data intelligently.
"""
            else:
                return jsonify({"error": "No band power data returned."}), 500

        # Other image types (visualizations)
        elif "image" in content_type:
            return jsonify({
                "error": "MCP returned an image. Cannot reason with image.",
                "hint": "Try a tool that returns JSON instead."
            }), 400

        else:
            try:
                mcp_data = mcp_res.json()
            except ValueError:
                return jsonify({"error": "Invalid JSON from MCP", "raw": mcp_res.text}), 500

            if tool_endpoint == "/filter-edf":
                summary_data = summarize_filtered_data(mcp_data.get("filtered_data", {}))
                final_prompt = f"""
Context:
{rag_context}

User Question:
{question}

MCP Filtered EEG Summary:
{summary_data}

Answer based on signal stats and context.
"""
            else:
                final_prompt = f"""
Context:
{rag_context}

User Question:
{question}

MCP Data:
{mcp_data}

Answer clearly using domain knowledge.
"""

    else:
        # No file-based tool, just a question
        final_prompt = f"""
Context:
{rag_context}

User Question:
{question}

Answer clearly using the EEG domain knowledge above.
"""

    answer = llm.invoke(final_prompt)
    return jsonify({"status": "success", "answer": answer})

if __name__ == '__main__':
    app.run(port=5002)
