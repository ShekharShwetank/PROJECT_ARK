import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer

class RAGUtils:
    def __init__(self, db_path="db", llm_model="mistral", embedding_model="all-mpnet-base-v2"):
        print("Initializing RAG utilities...")
        self.llm = OllamaLLM(model=llm_model)
        self.embedding_model = SentenceTransformer(embedding_model, device='cuda')
        self.db_client = chromadb.PersistentClient(path=db_path)
        print("RAG utilities initialized.")

    def create_rag_chain(self, collection_name, prompt_template):
        """Create a RAG chain for a specific collection."""
        try:
            collection = self.db_client.get_collection(name=collection_name)
        except ValueError:
            raise ValueError(f"Collection '{collection_name}' does not exist. Please ingest data first using ingest.py")

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

# Global instance
_rag_utils_instance = RAGUtils()
