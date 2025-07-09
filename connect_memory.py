from utils import get_embedding_model
from utils import load_FAISS_vectorstore
from utils import get_retriever
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from PM_prompt_template import load_prompt_template



from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough

# Initialize
load_dotenv()
prompt = load_prompt_template()
parser = StrOutputParser()

# 1. Setup Embeddings and Retriever
model = get_embedding_model() # sentence transformer
vectorstore = load_FAISS_vectorstore(model) 
retrievers = get_retriever(model)

from utils import load_llm
model = load_llm()

# Assume:
# - `retriever` fetches relevant context (e.g., from a vectorstore)
# - `prompt` is your improved PromptTemplate
# - `model` is your LLM (e.g., ChatOpenAI)
# - `parser` parses the output (e.g., StrOutputParser)

rag_chain = (
    RunnableParallel({
        "context": retrievers,  # Retrieves documents
        "question": RunnablePassthrough()  # Passes the user's question directly
    })
    | {
        # Map retrieved context/question to ALL prompt variables
    "knowledge_area": lambda x: x["question"], # Assuming the knowledge area is directly in the question
    "length_input": lambda x: "detailed, brief", # This could be extracted from question or set based on context length
    "style_input": lambda x: "Technical", # This could be extracted from question or inferred
    "focus": lambda x: "defining and controlling project scope", # This would ideally be extracted from x["context"] based on the knowledge_area
    "key_processes": lambda x: "Plan Scope Management, Collect Requirements, Define Scope, Create WBS, Validate Scope, Control Scope", # This would ideally be extracted from x["context"]
    "outputs": lambda x: "Scope Baseline, Requirements Documentation, WBS", # This would ideally be extracted from x["context"]
    }
    | prompt  # Your improved PromptTemplate
    | model
    | parser
)

# 3. Execute
result = rag_chain.invoke('Explain project Proc management according to PMBOK. make it Action-Oriented')
print(result)

# Visualization
#result.get_graph().print_ascii()