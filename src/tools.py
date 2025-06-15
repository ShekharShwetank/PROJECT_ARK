# src/tools.py

import subprocess
import os
import chromadb
import json
from langchain.tools import tool
from typing import List, Dict, Optional

# --- System Information Tools ---
@tool
def get_system_spec(spec_names: Optional[List[str]] = None) -> Dict:
    """Get system specifications. Returns CPU info by default."""
    try:
        client = chromadb.PersistentClient(path="db")
        collection = client.get_collection(name="ark_system_knowledge")
        profile_doc = collection.get(ids=["system_profile_doc_01"], include=["metadatas"])
        
        if not profile_doc or not profile_doc['metadatas']:
            return {"error": "System profile not found in ChromaDB"}
            
        metadata = profile_doc['metadatas'][0]
        
        # Default to CPU info if no specific fields requested
        if not spec_names:
            spec_names = ["cpu_model", "cpu_cores"]
            
        return {k: metadata.get(k, "Not available") for k in spec_names}
            
    except Exception as e:
        return {"error": f"Failed to get system specs: {str(e)}"}

# [Keep all other tools exactly the same as in previous version]
@tool
def get_disk_usage() -> Dict:
    """Get disk usage statistics"""
    try:
        result = subprocess.run(['df', '-h'], capture_output=True, text=True, check=True)
        return {"disk_usage": result.stdout}
    except Exception as e:
        return {"error": str(e)}

# --- File System Tools ---
@tool
def list_files(directory: str = ".") -> Dict:
    """List files in a directory (defaults to current directory)"""
    try:
        return {"files": os.listdir(directory)}
    except Exception as e:
        return {"error": str(e)}

@tool
def read_file_content(file_path: str) -> Dict:
    """Read file content"""
    try:
        with open(file_path, 'r') as f:
            return {"content": f.read()}
    except Exception as e:
        return {"error": str(e)}

@tool
def create_file(file_path: str, content: str) -> Dict:
    """Create a new file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return {"status": "File created", "path": file_path}
    except Exception as e:
        return {"error": str(e)}

# --- Process Management Tools ---
@tool
def list_running_processes(process_name: str = "") -> Dict:
    """List running processes (optionally filtered by name)"""
    try:
        if process_name:
            cmd = ['pgrep', '-fl', process_name]
        else:
            cmd = ['ps', 'aux']
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {"processes": result.stdout.splitlines()}
    except Exception as e:
        return {"error": str(e)}

# --- Project Tools ---
@tool
def create_kicad_project(project_name: str, components: List[str]) -> Dict:
    """Create a KiCad project with specified components"""
    try:
        project_dir = f"projects/{project_name}"
        os.makedirs(project_dir, exist_ok=True)
        
        with open(f"{project_dir}/{project_name}.pro", 'w') as f:
            f.write("(kicad_project)\n")
            
        with open(f"{project_dir}/{project_name}.sch", 'w') as f:
            f.write("EESchema Schematic File Version 4\n")
            f.write("LIBS:power,device,conn,linear,regul,switch,transf\n")
            
            # Add basic components
            for i, component in enumerate(components, 1):
                f.write(f"F {i} \"{component}\" H 2000 2000 50  0001 C CNN\n")
                f.write(f"P 2000 2000\n")
        
        return {
            "status": "Project created",
            "directory": project_dir,
            "files": [f"{project_name}.pro", f"{project_name}.sch"],
            "components": components
        }
    except Exception as e:
        return {"error": str(e)}