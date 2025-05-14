from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Example context data (Replace with EEG/MCP docs)
docs = [
    Document(page_content="EEG filtering removes noise from brain signals."),
    Document(page_content="Bandpass filtering helps isolate EEG frequency bands."),
    Document(page_content="EEG signals are visualized for medical diagnosis."),
    Document(page_content="Resume Parsing extracts structured data from resumes."),
    Document(page_content="Candidate matching compares resume skills to job descriptions.")
]

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(splits, embeddings, persist_directory="./vectorstore")
db.persist()

print("✅ Vector DB populated.")
