import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys

# --- CONFIGURATION ---
DB_PATH = "db"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
COLLECTION_NAME = "ark_system_knowledge"
LLM_MODEL_NAME = "mistral"

def main():
    """
    Main function to run the ARK assistant's query engine.
    """
    print("--- Initializing ARK Assistant ---")

    # --- 1. Initialize Vector Database and Retriever ---
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')

    print("Connecting to vector database...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    def retrieve_context(query_text, n_results=5):
        """
        Retrieves relevant context from the vector database.
        """
        query_embedding = embedding_model.encode(query_text).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return "\n---\n".join(results['documents'][0])

    # --- 2. Initialize the LLM ---
    print(f"Initializing LLM: {LLM_MODEL_NAME}...")
    llm = Ollama(model=LLM_MODEL_NAME)
    print("LLM Initialized.")

    # --- 3. Define the RAG Prompt Template ---
    template = """
    You are ARK, a helpful AI assistant with access to specific information about this Ubuntu system.
    Your task is to answer the user's question based *only* on the context provided below.
    If the context does not contain the answer, state that the information is not available in your knowledge base.
    Do not make up information. Be concise and accurate.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # --- 4. Construct the RAG Chain ---
    rag_chain = (
        {"context": (lambda x: retrieve_context(x["question"])), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- ARK is ready. Ask your questions. ---")
    print("Type 'exit' or 'quit' to end the session.")

    # --- 5. Interactive Query Loop with STREAMING ---
    while True:
        try:
            question = input("\n> You: ")
            if question.lower() in ['exit', 'quit']:
                print("\nARK shutting down. Goodbye.")
                break

            print("\n> ARK: ", end="", flush=True)

            # --- MODIFICATION: Use .stream() instead of .invoke() ---
            full_response = ""
            for chunk in rag_chain.stream({"question": question}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print() # Add a newline at the end of the full response

        except KeyboardInterrupt:
            print("\n\nARK shutting down. Goodbye.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
