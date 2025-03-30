# Advanced RAG Techniques

This folder contains examples demonstrating advanced Retrieval Augmented Generation (RAG) techniques that go beyond basic implementations.

## Files

- `ex27_mapreduce_langraph_summary.py`: Map-reduce pattern for document summarization using LangGraph
- `ex28_parent_RAG.py`: Parent-child RAG architecture for improved context handling
- `ex29.py`: Advanced RAG implementation with additional features
- `ex30.py`: Experimental RAG techniques with blog tutorial

## Key Concepts

- Map-reduce pattern for processing large documents
- Parent-child architectures for hierarchical retrieval
- Multi-query retrieval strategies
- Query transformation and rewriting
- Hybrid search techniques
- Self-reflection and correction in RAG systems
- Evaluation and metrics for RAG performance

## Advanced Techniques Overview

These examples showcase cutting-edge approaches to RAG that address common limitations of basic implementations:

1. **Map-Reduce**: Breaking down large documents into manageable chunks, processing each independently, and then combining the results.

2. **Parent-Child Architecture**: Using a hierarchical approach where a parent retriever finds relevant document sections, and child retrievers focus on specific details.

3. **Query Transformation**: Improving retrieval by rewriting or expanding the original query to better match relevant documents.

4. **Hybrid Search**: Combining different search techniques (semantic, keyword, metadata) for more comprehensive retrieval.

5. **Self-Reflection**: Implementing mechanisms for the system to evaluate and improve its own responses.

## Usage

```bash
python ex27_mapreduce_langraph_summary.py
```

These advanced techniques can significantly improve the quality, relevance, and accuracy of RAG systems, especially for complex use cases involving large document collections or specialized domains.
