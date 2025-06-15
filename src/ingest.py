# src/ingest.py

import json
import chromadb

def main():
    """Ingests the structured system_profile.json into ChromaDB."""
    print("--- Starting Structured Knowledge Ingestion ---")
    
    db_path = "db"
    collection_name = "ark_system_knowledge"
    profile_path = "data/system_profile/system_profile.json"
    
    # 1. Load the structured data
    try:
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Profile not found at {profile_path}. Please run acquire_data.py first.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {profile_path}.")
        return

    # 2. Connect to ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    # 3. Prepare document and metadata for ingestion
    # Create a single document summarizing the system.
    document_content = f"This document contains the system profile. CPU is {profile_data.get('cpu_info', {}).get('model_name')}. Total memory is {profile_data.get('memory_info', {}).get('total_memory')}."
    
    # The metadata will be a flattened version of the JSON for storage.
    metadata = {
        "doc_type": "system_profile",
        "cpu_model": profile_data.get("cpu_info", {}).get("model_name"),
        "cpu_cores": profile_data.get("cpu_info", {}).get("cpu_cores"),
        "total_memory": profile_data.get("memory_info", {}).get("total_memory"),
        # Convert list of GPUs to a single string
        "gpu_models": ", ".join(profile_data.get("gpu_info", [])), 
    }
    
    # Add other top-level info as well
    metadata['kernel_info'] = profile_data.get('kernel_info')

    # 4. Add to the collection
    collection.add(
        ids=["system_profile_doc_01"],
        documents=[document_content],
        metadatas=[metadata]
    )

    print(f"Successfully ingested system profile into collection '{collection_name}'.")
    print(f"Total documents in collection: {collection.count()}")
    print("--- Ingestion Finished ---")

if __name__ == "__main__":
    main()
