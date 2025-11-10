# Implementation Plan

## Overview
Thoroughly integrate the files in the `src/` folder by eliminating code duplication, improving path handling for directory queries, creating a modular architecture, and adding enhanced file system inspection tools. The main issues are: (1) duplicated RAG logic between `ask_ark.py` and `tools.py`, (2) incorrect path interpretation in the agent for personal directories, (3) lack of shared utilities, and (4) limited file inspection capabilities.

## Types
No new types or data structures are needed. Existing classes like `RAGTools` will be refactored into a shared module.

## Files
### New Files
- `src/rag_utils.py`: Shared RAG utilities to eliminate duplication between `ask_ark.py` and `tools.py`.
- `src/common.py`: Common functions for path handling and utilities used across modules.

### Modified Files
- `src/tools.py`: Remove duplicated RAG logic, import from `rag_utils.py`, add path normalization in `list_directory`, add new file inspection tools (`run_shell_command`, `get_directory_size`).
- `src/ask_ark.py`: Refactor to use shared `rag_utils.py` instead of inline RAG code.
- `src/run_agent.py`: Enhance prompt for better path interpretation, add path preprocessing in `execute_tool`, add new tools to TOOLS_MAP.
- `src/ingest.py`: No changes needed.
- `src/delete_from_ark.py`: No changes needed.

### Deleted Files
None.

## Functions
### New Functions
- `rag_utils.py`: `create_rag_chain(collection_name, prompt_template)`, `retrieve_context(query_text, collection, n_results=10)`
- `common.py`: `normalize_path(path)`, `expand_home_path(path)`
- `tools.py`: `run_shell_command(command)`, `get_directory_size(path)`

### Modified Functions
- `tools.py` `list_directory`: Add path normalization before expansion.
- `run_agent.py` `execute_tool`: Preprocess paths for `list_directory` to handle common directory names, add handling for new tools.

### Removed Functions
None.

## Classes
### New Classes
- `RAGUtils` in `rag_utils.py`: Singleton class for RAG operations.

### Modified Classes
- `RAGTools` in `tools.py`: Remove and replace with import from `rag_utils.py`.

### Removed Classes
- `RAGTools` from `tools.py`.

## Dependencies
No new dependencies. Existing ones remain: langchain, chromadb, sentence-transformers, ollama.

## Testing
Test the integrated system by running the agent and querying directories like "application documents folder" to ensure correct path resolution. Test RAG queries for both system and project knowledge. Verify ask_ark.py still works independently. Test new file inspection tools like `run_shell_command` for commands like `tree`, `grep`, etc., and `get_directory_size` for directory size queries.

## Implementation Order
1. Create `src/common.py` with path utilities.
2. Create `src/rag_utils.py` with shared RAG logic.
3. Refactor `src/tools.py` to use new modules and add new file inspection tools.
4. Refactor `src/ask_ark.py` to use `rag_utils.py`.
5. Update `src/run_agent.py` for better path handling and add new tools.
6. Test the integration.
