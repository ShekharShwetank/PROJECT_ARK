import subprocess
from langchain.tools import tool
import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer

# --- Reusable Components ---
# Moved these here to avoid re-initializing them for every tool call.
# This is a key optimization.

class RAGTools:
    def __init__(self):
        print("Initializing RAG tool components...")
        self.llm = OllamaLLM(model="mistral")
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
        self.db_client = chromadb.PersistentClient(path="db")
        print("RAG tool components initialized.")

    def _create_rag_chain(self, collection_name, prompt_template):
        """Helper function to create a RAG chain for a specific collection."""
        collection = self.db_client.get_collection(name=collection_name)

        def retrieve_context(query_text, n_results=10):
            query_embedding = self.embedding_model.encode(query_text).tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
            context_with_metadata = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                source = meta.get('source', 'Unknown source')
                context_with_metadata.append(f"--- CONTEXT FROM: {source} ---\n{doc}")
            return "\n\n".join(context_with_metadata)

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        return (
            {"context": (lambda x: retrieve_context(x["question"])), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

# Initialize the components once
_rag_tools_instance = RAGTools()

# --- Tool Definitions ---

@tool
def get_disk_usage():
    """
    Returns the current disk usage of the system by running the 'df -h' command.
    This tool is useful for questions about real-time storage space, available disk, 
    or filesystem usage. It does not take any arguments.
    """
    print("\n>>> TOOL: Running 'df -h' command...")
    try:
        result = subprocess.run(['df', '-h'], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def query_system_knowledge(query: str) -> str:
    """
    Use this tool to answer questions about the computer's static hardware and software configuration, 
    such as CPU model, installed packages, kernel version, or memory size. 
    Input should be the user's full question.
    """
    print(f"\n>>> TOOL: Querying 'ark_system_knowledge' with: '{query}'")
    prompt_template = """
    Based *only* on the following context about system configuration, answer the question.
    If the context does not contain the answer, state that the information is not available.
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:
    """
    chain = _rag_tools_instance._create_rag_chain("ark_system_knowledge", prompt_template)
    return chain.invoke({"question": query})

@tool
def query_project_knowledge(query: str) -> str:
    """
    Use this tool to answer technical questions about specific software project files, source code, 
    functions, or documentation within the user's workspace. 
    Input should be the user's full question.
    """
    print(f"\n>>> TOOL: Querying 'ark_project_knowledge' with: '{query}'")
    prompt_template = """
    Based *only* on the following context from project source code and documents, answer the question.
    If the context does not contain the answer, state that the information is not available.
    When referencing code, mention the source file from the context.
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:
    """
    chain = _rag_tools_instance._create_rag_chain("ark_project_knowledge", prompt_template)
    return chain.invoke({"question": query})
