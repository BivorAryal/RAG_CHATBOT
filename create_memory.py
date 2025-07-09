from utils import load_pdf
from utils import create_chucks
from utils import get_embedding_model
from utils import create_faiss_vectorstore

# Step1. load and process documents 
documents = load_pdf() 
print(f'Loaded {len(documents)} documents')

# Step2: Split into chunks
text_chunks = create_chucks(documents)
print(f'Created {len(text_chunks)} chunks')

# Step3: Create Vector embeddings with documents
embedding_model = get_embedding_model() # sentence transformer
test_embedding = embedding_model.embed_query(text_chunks[0].page_content)
print(f'Embedding size: {len(test_embedding)}')

# Step4: # Create and save vector store (FAISS)
db = create_faiss_vectorstore(text_chunks, embedding_model)
print(f"""
Document Processing Complete:
- Pages loaded: {len(documents)}
- Text chunks: {len(text_chunks)} (sample: {len(text_chunks[0].page_content)} chars)
- Embedding dims: {len(test_embedding)} (sample: {test_embedding[:5]})
- Vectors stored: {db.index.ntotal}
""")
