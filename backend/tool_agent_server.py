from flask import Flask, request, jsonify, Response, send_file
import requests
from langchain_community.llms import Ollama
from vectorstore import retrieve_context
import numpy as np
import ast
import re  # For newline cleanup
import os
import uuid
import json

app = Flask(__name__)
llm = Ollama(model="mistral")

# 🔁 Step 1: Define keyword-based tool selector
TOOL_MAPPING = {
    # Read EDF metadata
    "read": "/read-edf",
    "channel": "/read-edf",
    "preview": "/read-edf",
    "sampling": "/read-edf",
    "shape": "/read-edf",

    # Visualize EEG waveform
    "visualize": "/visualize-edf",
    "plot": "/visualize-edf",
    "graph": "/visualize-edf",
    "waveform": "/visualize-edf",
    "draw": "/visualize-edf",

    # Power Spectrum (PSD)
    "psd": "/psd-edf",
    "spectrum": "/psd-edf",
    "spectral": "/psd-edf",
    "frequency": "/psd-edf",
    "band power": "/psd-edf",

    # Brainwave Band Features
    "feature": "/features-edf",
    "features": "/features-edf",
    "brainwave": "/features-edf",
    "delta": "/features-edf",
    "alpha": "/features-edf",
    "beta": "/features-edf",
    "gamma": "/features-edf",
    "theta": "/features-edf",
    "band": "/features-edf",
    "power": "/features-edf",

    # Filtering
    "filter": "/filter-edf",
    "clean": "/filter-edf",
    "noise": "/filter-edf",
    "denoise": "/filter-edf",
    "bandpass": "/filter-edf",

    # Summary Statistics
    "summary": "/summary-edf",
    "describe": "/summary-edf",
    "overview": "/summary-edf",
    "min": "/summary-edf",
    "max": "/summary-edf",
    "mean": "/summary-edf",
    "std": "/summary-edf",
    "range": "/summary-edf",

    # Export (future)
    "export": "/export-edf",
    "download": "/export-edf"
}

# Step 2: Find tool endpoint based on question
def pick_tool(question):
    for keyword, endpoint in TOOL_MAPPING.items():
        if keyword in question.lower():
            return endpoint
    return None

# Step 3: Summarize EEG signal for filter-edf responses
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

# Step 4: ToolAgent route to handle file + question
@app.route("/mcp/agent", methods=["POST"])
def agent():
    file = request.files.get("file")
    question = request.form.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    # Tool + Context selection
    tool_endpoint = pick_tool(question)
    rag_context = retrieve_context(question)

    # File-based tools (e.g. PSD, Filter)
    if tool_endpoint and file:
        files = {"file": (file.filename, file.stream, file.mimetype)}
        mcp_res = requests.post(f"http://localhost:5000{tool_endpoint}", files=files)

        if mcp_res.status_code != 200:
            return jsonify({"error": f"Tool call failed at {tool_endpoint}"}), 500

        content_type = mcp_res.headers.get("Content-Type", "")

        # Handle PSD with band powers in header
        if tool_endpoint == "/psd-edf" and "image" in content_type:
            band_powers_header = mcp_res.headers.get("X-Band-Powers")
            if band_powers_header:
                try:
                    band_powers = ast.literal_eval(band_powers_header)
                except Exception:
                    band_powers = band_powers_header
                final_prompt = f"""
You are an EEG assistant. Do not return code unless explicitly asked.

Context:
{rag_context}

Question:
{question}

MCP Band Powers:
{band_powers}

Interpret this EEG data intelligently and concisely.
"""
            else:
                return jsonify({"error": "No band power data returned."}), 500

        # Enhanced: Handle image-only responses from /visualize-edf
        elif "image" in content_type and tool_endpoint == "/visualize-edf":
            image_path = f"./uploads/{uuid.uuid4().hex}.png"
            os.makedirs("./uploads", exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(mcp_res.content)

            image_url = f"http://localhost:5000/uploads/{os.path.basename(image_path)}"

            final_prompt = f"""
You are an EEG assistant.

The user requested to visualize an EEG file. The waveform has been successfully generated.

Context:
{rag_context}

Question:
{question}

Note: The image was saved to {image_url}.
Do not describe the image, but explain what EEG waveforms generally represent.
"""

            answer = llm.invoke(final_prompt).strip()
            try:
                answer = json.loads(f'"{answer}"')
            except json.JSONDecodeError:
                pass
            answer = re.sub(r"[\n\r]+", " ", answer)
            answer = re.sub(r"\s{2,}", " ", answer).strip()

            return jsonify({
                "status": "success",
                "message": answer,
                "image_url": image_url
            })

        # Ignore other unknown image responses
        elif "image" in content_type:
            return jsonify({
                "error": "MCP returned an image. Cannot reason with image.",
                "hint": "Try a tool that returns JSON instead."
            }), 400

        # Handle all other JSON-based MCP tools
        else:
            try:
                mcp_data = mcp_res.json()
            except ValueError:
                return jsonify({"error": "Invalid JSON from MCP", "raw": mcp_res.text}), 500

            # For Filtered EEG
            if tool_endpoint == "/filter-edf":
                summary_data = summarize_filtered_data(mcp_data.get("filtered_data", {}))
                final_prompt = f"""
You are an EEG assistant. Avoid Python code in the response unless requested.

Context:
{rag_context}

Question:
{question}

MCP Filtered EEG Summary:
{summary_data}

Provide a meaningful interpretation for the user.
"""
            else:
                # General EEG tool output
                final_prompt = f"""
You are a helpful EEG assistant. Keep responses user-friendly and concise.

Context:
{rag_context}

Question:
{question}

MCP Data:
{mcp_data}

Answer clearly using EEG domain knowledge.
"""

    # No file tool — pure question-answering
    else:
        final_prompt = f"""
You are a helpful EEG assistant. Use the domain context below.

Context:
{rag_context}

Question:
{question}

Answer using your knowledge base. Avoid unnecessary technicalities.
"""

    # Step 5: Generate and clean LLM output
    answer = llm.invoke(final_prompt).strip()
    try:
        answer = json.loads(f'"{answer}"')
    except json.JSONDecodeError:
        pass
    answer = re.sub(r"[\n\r]+", " ", answer)
    answer = re.sub(r"\s{2,}", " ", answer).strip()

    # Optional format: plain text or JSON
    if request.args.get("format") == "raw":
        return Response(answer, mimetype="text/plain")
    else:
        return jsonify({"status": "success", "answer": answer})

# Serve downloadable EEG plots
@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_file(os.path.join("uploads", filename), mimetype="image/png")

if __name__ == '__main__':
    app.run(port=5002)
