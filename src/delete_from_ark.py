import argparse
import chromadb
import os

# --- CONFIGURATION ---
DB_PATH = "db"
BATCH_SIZE = 500 # The number of documents to delete in each batch

def main(collection_name: str, source_path: str):
    """
    Deletes documents from a specified collection that originate from a given source path.
    This version fetches all metadata, filters in Python, and deletes in batches.

    :param collection_name: The name of the collection to modify.
    :param source_path: The source directory path to delete.
    """
    print(f"--- Starting Deletion Process for collection: '{collection_name}' ---")
    print(f"Target source path: '{source_path}'")

    # --- 1. Connect to the Vector Database ---
    if not os.path.exists(DB_PATH):
        print(f"Error: Database path '{DB_PATH}' not found. Nothing to delete.")
        return

    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        print(f"Error: Collection '{collection_name}' not found. Cannot delete.")
        return

    # --- 2. Fetch all metadata and filter in Python ---
    print("Fetching all document metadata to filter for deletion...")
    # Get all documents with their metadata. This can be memory-intensive for huge dbs.
    all_docs = collection.get(include=["metadatas"])

    # Make sure the path is treated as a directory
    if not source_path.endswith(os.path.sep):
        source_path += os.path.sep

    ids_to_delete = []
    for i, metadata in enumerate(all_docs['metadatas']):
        # Check if the 'source' key exists and starts with the target path
        if metadata and 'source' in metadata and metadata['source'].startswith(source_path):
            ids_to_delete.append(all_docs['ids'][i])

    if not ids_to_delete:
        print("No documents found matching the specified source path. Nothing to delete.")
        print("--- Deletion Process Finished ---")
        return

    total_to_delete = len(ids_to_delete)
    print(f"Found {total_to_delete} document chunks to delete.")

    # --- 3. Delete the Documents in Batches ---
    print(f"Deleting documents in batches of {BATCH_SIZE}...")

    deleted_count = 0
    for i in range(0, total_to_delete, BATCH_SIZE):
        batch_ids = ids_to_delete[i:i + BATCH_SIZE]
        collection.delete(ids=batch_ids)
        deleted_count += len(batch_ids)
        print(f"  - Deleted batch {i//BATCH_SIZE + 1}/{(total_to_delete + BATCH_SIZE - 1)//BATCH_SIZE}... ({deleted_count}/{total_to_delete})")

    print(f"\nSuccessfully deleted {total_to_delete} documents.")
    print(f"Current total documents in collection: {collection.count()}")
    print("--- Deletion Process Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete documents from ARK's knowledge base based on source path.")
    parser.add_argument("--path", type=str, required=True, help="The source directory path of the documents to delete.")
    parser.add_argument("--collection", type=str, required=True, help="The name of the ChromaDB collection to modify.")

    args = parser.parse_args()

    main(collection_name=args.collection, source_path=args.path)
