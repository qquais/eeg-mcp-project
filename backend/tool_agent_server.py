# controller_agent.py
from flask import Flask, request, jsonify
import requests
from langchain_community.llms import Ollama
import numpy as np

app = Flask(__name__)

# 🔁 Step 1: Set up local LLM
llm = Ollama(model="mistral")

# 🔁 Step 2: Define simple keyword-based tool selector

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
    "overview": "/summary-edf"
}


# 🔁 Step 3: Decide which tool to invoke based on question
def pick_tool(question):
    for keyword, endpoint in TOOL_MAPPING.items():
        if keyword in question.lower():
            return endpoint
    return None

# 🔁 Step 4: Summarize filtered EEG data
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

# 🔁 Step 5: Agent Endpoint
@app.route("/mcp/agent", methods=["POST"])
def agent():
    file = request.files.get("file")
    question = request.form.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    tool_endpoint = pick_tool(question)

    if tool_endpoint and file:
        files = {"file": (file.filename, file.stream, file.mimetype)}
        mcp_res = requests.post(f"http://localhost:5000{tool_endpoint}", files=files)

        if mcp_res.status_code != 200:
            return jsonify({"error": f"Tool call failed at {tool_endpoint}"}), 500

        content_type = mcp_res.headers.get("Content-Type", "")

        # Handle image/binary responses
        if "image" in content_type:
            return jsonify({
                "error": "MCP returned binary (image) data. Cannot pass to LLM.",
                "hint": "This endpoint returns a graph (PNG), not usable for text reasoning."
            }), 400

        try:
            mcp_data = mcp_res.json()
        except ValueError:
            return jsonify({"error": "Invalid JSON response from MCP server.", "raw": mcp_res.text}), 500

        # Optional pre-processing if tool is filter-edf
        if tool_endpoint == "/filter-edf":
            summary_data = summarize_filtered_data(mcp_data.get("filtered_data", {}))
            final_prompt = f"Question: {question}\n\nEEG Summary Stats: {summary_data}\n\nGive a final answer based on the above."
        else:
            final_prompt = f"Question: {question}\n\nMCP Data: {mcp_data}\n\nGive a final answer based on the above."
    else:
        final_prompt = question

    answer = llm.invoke(final_prompt)
    return jsonify({"status": "success", "answer": answer})

if __name__ == '__main__':
    app.run(port=5002)
