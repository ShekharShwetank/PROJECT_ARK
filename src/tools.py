import subprocess
try:
    from langchain.tools import tool
except Exception:
    # Provide a no-op decorator when langchain is not installed so the module
    # can still be imported for testing or in minimal environments.
    def tool(func=None, **kwargs):
        if func is None:
            def _decorator(f):
                return f
            return _decorator
        return func

from rag_utils import _rag_utils_instance
from common import normalize_path

# --- Tool Definitions ---

@tool
def get_disk_usage():
    """
    Returns the disk usage of the system by running 'df -h'.
    Use this for questions about storage space, available disk, or filesystem usage.
    This is the PREFERRED tool for checking disk usage.
    It does not take any arguments.
    """
    print("\n>>> TOOL: Running 'df -h' command...")
    try:
        result = subprocess.run(['df', '-h'], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def get_ram_usage():
    """
    Returns the current RAM (memory) usage, total size, and free space by running 'free -h'.
    This is the PREFERRED tool for any questions about memory or RAM, including total size,
    current usage, or available space.
    It does not take any arguments.
    """
    print("\n>>> TOOL: Running 'free -h' command...")
    try:
        result = subprocess.run(['free', '-h'], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def get_cpu_usage():
    """
    Returns the current CPU usage of the system by running 'top -bn1'.
    Use this for questions about CPU load, idle percentage, or process activity.
    This is the PREFERRED tool for checking CPU usage.
    It does not take any arguments.
    """
    print("\n>>> TOOL: Running 'top -bn1' command...")
    try:
        result = subprocess.run(['top', '-bn1'], capture_output=True, text=True, check=True)
        # Return only the relevant header and CPU line for brevity
        return "\n".join(result.stdout.splitlines()[:3])
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def run_shell_command(command: str) -> str:
    """
    Executes a shell command. Use this ONLY when no other specific tool can accomplish the task.
    This is a powerful tool that gives full terminal access. Use it for general-purpose commands
    like 'ps aux', 'grep', or other system introspection.
    Input should be a valid shell command string.
    SAFETY: Be careful with destructive commands (rm, mv, etc.). Always confirm intent.
    """
    print(f"\n>>> TOOL: Running shell command: '{command}'")
    try:
        # Add a basic safety check for obviously destructive commands
        dangerous_patterns = ['rm -rf /', 'mkfs', 'dd if=', ':(){:|:&};:']
        if any(pattern in command for pattern in dangerous_patterns):
            return "Error: This command appears to be dangerous and has been blocked for safety."
        
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30  # 30 second timeout
        )
        
        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr}"
        
        if result.returncode != 0:
            output += f"\n[Exit Code]: {result.returncode}"
        
        return output if output else "Command executed successfully (no output)"
    
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {e}"

@tool
def get_directory_size(path: str) -> str:
    """
    Returns the total disk usage size of a specific directory by running 'du -sh'.
    Use this to find out how much space a single folder is taking up.
    This is the PREFERRED tool for checking a directory's size.
    Input should be the directory path (e.g., '~/Desktop' or '/home/user/Documents').
    """
    print(f"\n>>> TOOL: Running 'du -sh {path}' command...")
    try:
        import os
        expanded_path = os.path.expanduser(path)
        result = subprocess.run(['du', '-sh', expanded_path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: Could not get size of directory '{path}'. {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def list_directory(path: str) -> str:
    """
    Lists the contents of a directory by running 'ls -lh'.
    Use this to see what files and folders are in a specific directory path.
    This is the PREFERRED tool for listing directory contents.
    Input should be the full path to the directory (e.g., '/home/user/Desktop' or '~/Desktop').
    """
    print(f"\n>>> TOOL: Running 'ls -lh {path}' command...")
    try:
        # Normalize and expand path
        expanded_path = normalize_path(path)
        result = subprocess.run(['ls', '-lh', expanded_path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: Could not list directory '{path}'. {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def read_file(path: str) -> str:
    """
    Reads and returns the first 4000 characters of a text file's content.
    Use this to understand what a specific file is about after finding it with 'list_directory'.
    Input should be the full path to the file (e.g., '~/Documents/notes.txt').
    """
    print(f"\n>>> TOOL: Reading file: '{path}'")
    try:
        # Normalize and expand path
        full_path = normalize_path(path)

        # Basic check to avoid reading binary files
        import mimetypes
        mime_type, _ = mimetypes.guess_type(full_path)
        if mime_type and not mime_type.startswith('text/'):
            return f"Error: Cannot read file '{path}'. It appears to be a binary file ({mime_type})."

        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(4000) # Read only the first 4000 characters

        return content if content else "File is empty."
    except FileNotFoundError:
        return f"Error: File not found at '{path}'."
    except IsADirectoryError:
        return f"Error: Path '{path}' is a directory, not a file. Use 'list_directory' instead."
    except Exception as e:
        return f"An unexpected error occurred while reading file '{path}': {e}"

@tool
def query_system_knowledge(query: str) -> str:
    """
    Answers questions about system configuration, file paths, or hardware details by querying a knowledge base.
    This is the PREFERRED tool for finding the correct path to a user's directory (e.g., "where is the application documents folder?").
    Also use this for deep details like CPU models, kernel versions, or hardware specs from 'lshw'.
    Do NOT use this for real-time stats like current CPU/RAM usage.
    Input should be a clear question about the system.

    NOTE: This requires the 'ark_system_knowledge' collection to be created first.
    """
    print(f"\n>>> TOOL: Querying 'ark_system_knowledge' with: '{query}'")
    prompt_template = """
    Based *only* on the following context about system configuration, answer the question.
    If the context does not contain the answer, state that the information is not available.
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:
    """
    try:
        # If the query is about finding a path, use a more specific prompt.
        if any(kw in query.lower() for kw in ["path to", "find folder", "directory of", "where is"]):
            prompt_template = """
            Based ONLY on the following file system structure, find the full, correct path for the user's query.
            CONTEXT: {context}
            QUESTION: {question}
            ANSWER:"""

        chain = _rag_utils_instance.create_rag_chain("ark_system_knowledge", prompt_template)
        return chain.invoke({"question": query})
    except ValueError as e:
        # Provide a specific, actionable error message if the collection doesn't exist.
        if "does not exist" in str(e):
            # Return a structured, actionable error that the agent can parse and act on.
            return ("ACTION_REQUIRED: Collection 'ark_system_knowledge' not found. "
                    "Run the following commands sequentially: "
                    "1. `./src/scripts/gather_system_info.sh` then "
                    "2. `python3 src/ingest.py --path data/system_info --collection ark_system_knowledge`")
        return f"Error querying system knowledge: {str(e)}"

@tool
def query_project_knowledge(query: str) -> str:
    """
    Use this tool to answer technical questions about specific software project files, source code,
    functions, or documentation within the user's workspace.
    Input should be the user's full question.
    NOTE: This requires the 'ark_project_knowledge' collection to be created first.
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
    try:
        chain = _rag_utils_instance.create_rag_chain("ark_project_knowledge", prompt_template)
        return chain.invoke({"question": query})
    except ValueError as e:
        # Provide a specific, actionable error message if the collection doesn't exist.
        if "does not exist" in str(e):
            # Return a structured, actionable error.
            return ("ACTION_REQUIRED: Collection 'ark_project_knowledge' not found. "
                    "Run the following command: "
                    "`python3 src/ingest.py --path /path/to/your/project --collection ark_project_knowledge`")
        return f"Error querying project knowledge: {str(e)}"
