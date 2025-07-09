from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# Step 1: load and process documents
def load_pdf():  
    loader = DirectoryLoader(
        path='PM_BOOK',  
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Step2: Split into chunks
def create_chucks(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # character content
        chunk_overlap = 100, # tells you how many character will overlaps between 2 chunks
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
        )
    text_chunks= splitter.split_documents(extracted_data) # split document
    return text_chunks

# Step 3:Create Vector Embeddings
def get_embedding_model():
  embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'
  )
  return embedding_model

# Step 4: Create Vector Store embeddings in FAISS
DB_FAISS_PATH = "db_faiss"

def create_faiss_vectorstore(text_chunks, embedding_model):
    """Creates and saves FAISS vectorstore"""
    # Create and save in one step
    db = FAISS.from_documents(
        documents=text_chunks,
        embedding=embedding_model
    )
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Saved FAISS index to {DB_FAISS_PATH}")
    return db
################################
################################
    # PHASE 2 Connection

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# load component
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=150,
        temperature=1.2,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.2
        )
    model = ChatHuggingFace(llm=llm)
    return model

# Import prompt
from PM_prompt_template import load_prompt_template
# Get Template
template = load_prompt_template()
# Usage
prompt = template.format(
   length_input =  "brief, detailed",
   knowledge_area = "Scope Management, Risk Management",
   style_input = "Beginner-Friendly, Technical, Action-Oriented, Visual",
   focus = "defining and controlling project scope",
   key_processes = "Plan Scope Management, Collect Requirements, Define Scope, Create WBS, Validate Scope, Control Scope",
   outputs = "Scope Baseline, Requirements Documentation, WBS"
   )

# Step 2: Connect LLMs with FAISS (Connect with Vector Store and create CHAINS)
DB_FAISS_PATH="db_faiss"
def load_FAISS_vectorstore(embedding_model):
  return FAISS.load_local(
      DB_FAISS_PATH,
      embedding_model,
      allow_dangerous_deserialization=True  # Required for security
      )

def get_retriever (embedding_model):
    vectorstore = load_FAISS_vectorstore(embedding_model)
    return vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs={'k':3,
                       "lambda_mult": .2})