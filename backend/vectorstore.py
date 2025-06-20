import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Load EEG-related .txt docs
def load_documents():
    doc_dir = "./docs"  # put your .txt files here
    documents = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(doc_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

# Build and save FAISS vectorstore
def build_vectorstore():
    documents = load_documents()
    # Splits large documents into 500-character chunks with 50-character overlaps
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    # Initializes the embedding model 
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    # Converts chunks into embeddings and builds a FAISS vector index.
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local("vectorstore/db")
    print("✅ Vectorstore built and saved to vectorstore/db")

# Retrieve top-k, (here I took k = 2 ) similar chunks for a given question
def retrieve_context(question, k=2):
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    # Acknowledges risk from loading a pickle file by passing allow_dangerous_deserialization=True
    vectorstore = FAISS.load_local("vectorstore/db", embedding, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(question, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    build_vectorstore()
