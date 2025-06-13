import os
import time
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- CONFIGURATION ---
# This is the path to the directory containing the raw text files.
DATA_PATH = "data/system_info"
# This is the path where the vector database will be stored.
DB_PATH = "db"
# This is the name of the embedding model we will use. It's a powerful and widely used model.
# It will be downloaded automatically the first time you run the script.
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
# The name for our collection in the vector database.
COLLECTION_NAME = "ark_system_knowledge"

def main():
    """
    Main function to handle the ingestion process.
    """
    print("--- Starting ARK Knowledge Ingestion ---")

    # --- 1. Load Documents ---
    print(f"Step 1: Loading documents from '{DATA_PATH}'...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path '{DATA_PATH}' not found. Please run the gather_system_info.sh script first.")
        return

    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    if not documents:
        print("No documents found to ingest. Exiting.")
        return
    print(f"Successfully loaded {len(documents)} documents.")

    # --- 2. Split Documents into Chunks ---
    # We split the documents into smaller chunks to improve the relevance of search results.
    # This helps the LLM focus on more specific context.
    print("Step 2: Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The max number of characters in a chunk.
        chunk_overlap=200   # The number of characters to overlap between chunks.
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documents split into {len(chunks)} chunks.")

    # --- 3. Initialize Embedding Model ---
    # This will run on your GPU thanks to the PyTorch with CUDA installation.
    print(f"Step 3: Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    # device='cuda' explicitly tells the model to use the GPU.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')
    print("Embedding model loaded onto GPU.")

    # --- 4. Initialize Vector Database ---
    # We are using a persistent client, which saves the database to disk in the DB_PATH.
    print(f"Step 4: Initializing vector database at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Get or create the collection. This is like a table in a traditional database.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Using cosine distance for similarity
    )
    print(f"Collection '{COLLECTION_NAME}' is ready.")

    # --- 5. Generate Embeddings and Ingest into DB ---
    print("Step 5: Generating embeddings and ingesting into the database...")
    start_time = time.time()

    # We will ingest in batches for efficiency
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Extract the text content from each chunk document
        batch_texts = [doc.page_content for doc in batch]

        # Generate embeddings for the batch of texts
        embeddings = embedding_model.encode(batch_texts, show_progress_bar=False).tolist()

        # Prepare IDs and metadata for the batch
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        metadatas = [doc.metadata for doc in batch]

        # Add the batch to the collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=metadatas
        )
        print(f"  - Ingested batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")

    end_time = time.time()
    print(f"\nIngestion complete. Took {end_time - start_time:.2f} seconds.")
    print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")
    print("--- ARK Knowledge Ingestion Finished ---")

if __name__ == "__main__":
    main()
