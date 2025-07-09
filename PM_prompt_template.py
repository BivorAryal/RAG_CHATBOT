from langchain_core.prompts import PromptTemplate

def load_prompt_template():
    prompt = PromptTemplate(
        template = """
Generate a {length_input} summary of the PMBOK knowledge area "{knowledge_area}" in a {style_input} style.

**Core Focus:**  {focus}

**Key Processes:**  {key_processes}

**Critical Outputs:**  {outputs}

**Special Instructions:**  
- For "Beginner-Friendly": Use analogies (e.g., "Scope is like building a house blueprint")  
- For "Technical": Include PMBOK process group alignments  
- For "Action-Oriented": Add bullet points for immediate implementation  
- For "Visual": Suggest charts/diagrams (e.g., WBS for Scope Management)  

Avoid jargon if style is "Beginner-Friendly".
""",
    input_variables=["book_title", "style_input", "length_input", "methodologies", "tools", "audience"],
    )
    return prompt