from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load all text files from docs/ folder
loader = DirectoryLoader('./docs', glob="**/*.txt")
documents = loader.load()

# Create Ollama embeddings (local)
embeddings = OllamaEmbeddings(model="mistral")

# Populate Vectorstore with MCP docs
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./vectorstore"
)

print("✅ Vectorstore populated with MCP documentation.")
