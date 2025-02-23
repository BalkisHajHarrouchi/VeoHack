from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import langchain
import os
import shutil

langchain.debug = True

# ✅ Set Hugging Face cache directory
os.environ["HF_HOME"] = "C:/Users/MSI/.cache/huggingface"

# ✅ Load the embedding model (LaJavaness with 1024 dimensions)
EMBEDDING_MODEL = "Lajavaness/bilingual-embedding-large"
model_kwargs = {"trust_remote_code": True}
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

# ✅ Ensure ChromaDB is reset
chroma_db_path = "emb/"
if os.path.exists(chroma_db_path):
    print("🛑 Deleting old ChromaDB to prevent conflicts...")
    shutil.rmtree(chroma_db_path)  # Delete existing database

# ✅ Create directories if not exist
os.makedirs("emb", exist_ok=True)

# ✅ Load and split job offers text file
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="\n\n")
loader_jobs = TextLoader("jobs.txt")  # Load jobs from text file
docs_jobs = loader_jobs.load_and_split(text_splitter=text_splitter)


# ✅ Initialize ChromaDB
db_jobs = Chroma(persist_directory="emb/jobsDB", embedding_function=embedding_function)

# ✅ Add text data to ChromaDB
db_jobs.add_texts(texts=[doc.page_content for doc in docs_jobs])

print("✅ Jobs and candidates stored successfully in ChromaDB as text!")
