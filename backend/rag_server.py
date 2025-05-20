from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify

# ✅ Load Local Vector DB
embeddings = OllamaEmbeddings(model='mistral')
db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)

# ✅ Local LLM (Mistral via Ollama)
llm = Ollama(model='mistral')

# ✅ RetrievalQA Chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# ✅ Flask App
app = Flask(__name__)

# @app.route('/query', methods=['POST'])
# def query():
#     data = request.get_json()
#     question = data.get('question', '')
#     if not question:
#         return jsonify({'error': 'Question is required'}), 400

#     answer = qa.run(question)
#     return jsonify({'answer': answer})

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'status': 'error', 'message': 'Question is required'}), 400

        answer = qa.run(question)
        return jsonify({'status': 'success', 'answer': answer.strip()})

    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
