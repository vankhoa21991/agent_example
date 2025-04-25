#!/usr/bin/env python
"""
Example script to demonstrate using FastEmbed with Qdrant for RAG
Load documents from files and perform RAG queries
"""

import argparse
import os
import glob
from typing import List, Dict, Optional
import sys
from langchain_core.documents import Document
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)

# Import our RAG implementation
from fastemb_qdrant_rag import FastEmbedQdrantRAG

def load_document(file_path: str) -> List[Document]:
    """Load a document from a file path"""
    _, ext = os.path.splitext(file_path)
    
    # Select the appropriate loader based on file extension
    if ext.lower() == '.txt':
        loader = TextLoader(file_path)
    elif ext.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext.lower() in ['.md', '.markdown']:
        loader = UnstructuredMarkdownLoader(file_path)
    elif ext.lower() in ['.html', '.htm']:
        loader = UnstructuredHTMLLoader(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return []
    
    try:
        return loader.load()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FastEmbed + Qdrant RAG Example"
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        help="Directory containing documents to index",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.txt,*.pdf,*.md,*.html",
        help="File patterns to load (comma-separated)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run against the indexed documents",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create RAG system
    rag = FastEmbedQdrantRAG()
    
    # Load and index documents if a directory is provided
    if args.docs_dir:
        if not os.path.isdir(args.docs_dir):
            print(f"Error: {args.docs_dir} is not a valid directory")
            sys.exit(1)
        
        # Get all files matching the pattern
        file_patterns = args.file_pattern.split(',')
        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(os.path.join(args.docs_dir, pattern.strip())))
        
        if not all_files:
            print(f"No files found matching pattern {args.file_pattern} in {args.docs_dir}")
            sys.exit(1)
        
        print(f"Found {len(all_files)} files to process")
        
        # Load all documents
        all_docs = []
        for file_path in all_files:
            print(f"Loading {file_path}")
            docs = load_document(file_path)
            if docs:
                # Add source filename to metadata
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(file_path)
                all_docs.extend(docs)
        
        print(f"Loaded {len(all_docs)} documents")
        
        # Add documents to RAG system
        if all_docs:
            rag.add_documents(all_docs)
            print("Documents indexed successfully")
    
    # Process a single query if provided
    if args.query:
        print(f"\nQuery: {args.query}")
        response = rag.query(args.query, k=args.top_k)
        print(f"\nResponse: {response}")
    
    # Interactive mode
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            response = rag.query(query, k=args.top_k)
            print(f"\nResponse: {response}")

if __name__ == "__main__":
    main() 