# Building a Document Processing and RAG System with OpenAI and LangGraph

In this tutorial, we'll explore how to build a powerful document processing and retrieval-augmented generation (RAG) system using OpenAI's language models and LangGraph. This system can process PDF files and URLs, convert them to markdown, and use them to answer questions or generate summaries.

The code is divided into two main parts:
1. **Data Ingestion**: Processing documents and storing them in a vector database
2. **Q/A and Summary with Agent**: Using a LangGraph workflow to process user queries

Let's dive into each part in detail.

## Part 1: Data Ingestion

The data ingestion process involves converting documents (PDFs or web pages) to markdown, splitting them into chunks, creating embeddings, and storing them in a vector database. Let's look at the key components:

### DocumentProcessor Class

The `DocumentProcessor` class handles the conversion of documents to markdown format:

```python
class DocumentProcessor:
    """Handles document processing with docling"""
    
    def __init__(self):
        """Initialize the document processor"""
        pass
        
    def process_document(self, source: str) -> Tuple[str, str]:
        """
        Process a document from a file path or URL
        
        Args:
            source: Path to a file or URL
            
        Returns:
            Tuple of (markdown_content, output_path)
        """
        logger.info(f"Processing document: {source}")
        
        try:
            # Determine if source is a URL or a file
            is_url = source.startswith(('http://', 'https://'))
            
            # Convert to markdown based on source type
            if is_url:
                markdown_content = url_to_markdown(source)
            else:
                markdown_content = pdf_to_markdown(source)
            
            # Generate output filename
            source_name = os.path.basename(source) if os.path.exists(source) else source.split('/')[-1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{source_name}_{timestamp}.md"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Save markdown to file
            with open(output_path, "w") as f:
                f.write(markdown_content)
                
            logger.info(f"Saved markdown to: {output_path}")
            
            return markdown_content, output_path
            
        except Exception as e:
            logger.error(f"Error processing document {source}: {str(e)}")
            raise
```

This class uses two helper functions:

1. `url_to_markdown`: Converts a web page to markdown using docling
2. `pdf_to_markdown`: Converts a PDF file to markdown using docling

Both functions use the `DocumentConverter` from the docling library, which provides powerful document processing capabilities, including OCR for images and table structure recognition.

### ChromaIndexer Class

The `ChromaIndexer` class handles the indexing of documents in ChromaDB, a vector database:

```python
class ChromaIndexer:
    """Handles document indexing in ChromaDB"""
    
    def __init__(self):
        """Initialize the ChromaDB indexer"""
        self.db_path = CHROMA_DB_PATH
        self.collection_name = COLLECTION_NAME
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=self.db_path,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        logger.info("Vector store initialized successfully")
```

The key methods of this class are:

1. `index_document`: Splits a document into chunks, creates embeddings, and stores them in ChromaDB:

```python
def index_document(self, content: str, metadata: Dict) -> List[str]:
    """
    Index a document in ChromaDB
    
    Args:
        content: Document content
        metadata: Document metadata
        
    Returns:
        List of document IDs
    """
    logger.info(f"Indexing document: {metadata.get('file_name', 'unknown')}")
    
    try:
        # Create a Document object
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        
        # Split the document
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ','],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents([doc])
        
        # Generate UUIDs for all chunks
        doc_ids = [str(uuid4()) for _ in range(len(chunks))]
        
        # Add documents to vector store
        self.vector_store.add_documents(documents=chunks, ids=doc_ids)
        
        # Persist the changes
        if hasattr(self.vector_store, '_collection'):
            self.vector_store.persist()
        
        logger.info(f"Indexed {len(chunks)} chunks")
        return doc_ids
        
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        # Return empty list in case of error
        return []
```

2. `search`: Retrieves relevant documents from ChromaDB based on a query:

```python
def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
    """
    Search for documents in ChromaDB
    
    Args:
        query: Query string
        top_k: Number of results to return
        
    Returns:
        List of document chunks
    """
    logger.info(f"Searching for: {query}")
    
    try:
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        # Search for documents
        docs = retriever.invoke(query)
        return docs
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        # Return empty list in case of error
        return []
```

### DocumentSystem Class

The `DocumentSystem` class integrates the `DocumentProcessor` and `ChromaIndexer` classes to provide a complete document processing system:

```python
class DocumentSystem:
    """Main system that integrates all components"""
    
    def __init__(self):
        """Initialize the document system"""
        self.processor = DocumentProcessor()
        self.indexer = ChromaIndexer()
        self.documents = {}  # Store processed documents
        
        # Create output directory if it doesn't exist
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # Load previously processed documents
        self._load_processed_documents()
```

The key methods of this class are:

1. `_load_processed_documents`: Loads previously processed documents from the output directory:

```python
def _load_processed_documents(self):
    """Load previously processed documents from the output directory and ChromaDB"""
    logger.info("Loading previously processed documents")
    
    try:
        # Get all markdown files in the output directory
        md_files = list(Path(OUTPUT_DIR).glob("*.md"))
        
        if not md_files:
            logger.info("No previously processed documents found")
            return
            
        # Load each markdown file
        for md_file in md_files:
            try:
                # Read the markdown file
                with open(md_file, "r") as f:
                    content = f.read()
                    
                # Create metadata
                metadata = {
                    "file_name": md_file.name,
                    "source": "loaded_from_disk",
                    "processed_time": md_file.stat().st_mtime
                }
                
                # Store document
                self.documents[str(md_file)] = {
                    "content": content,
                    "metadata": metadata,
                    "doc_ids": []  # We don't have the doc_ids, but that's okay for summary tasks
                }
                
                logger.info(f"Loaded document: {md_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading document {md_file}: {str(e)}")
        
        logger.info(f"Loaded {len(self.documents)} documents")
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
```

2. `process_documents`: Processes a list of documents:

```python
def process_documents(self, sources: List[Union[str, Dict]]) -> None:
    """
    Process a list of documents
    
    Args:
        sources: List of file paths, URLs, or dictionaries with source and type
    """
    logger.info(f"Processing {len(sources)} documents")
    
    for source_item in sources:
        try:
            # Handle both string sources and dictionary sources
            if isinstance(source_item, dict):
                source = source_item["source"]
            else:
                source = source_item
            
            # Process document
            content, path = self.processor.process_document(source)
            
            # Create metadata
            metadata = {
                "file_name": os.path.basename(path),
                "source": source,
                "processed_time": datetime.now().isoformat()
            }
            
            # Index document
            doc_ids = self.indexer.index_document(content, metadata)
            
            # Store document
            self.documents[path] = {
                "content": content,
                "metadata": metadata,
                "doc_ids": doc_ids
            }
            
            logger.info(f"Successfully processed and indexed: {source}")
            
        except Exception as e:
            if isinstance(source_item, dict):
                logger.error(f"Error processing document {source_item['source']}: {str(e)}")
            else:
                logger.error(f"Error processing document {source_item}: {str(e)}")
```

### Data Ingestion Flow

The data ingestion flow can be summarized as follows:

1. The user provides a list of document sources (PDF files or URLs)
2. The `DocumentSystem` processes each document using the `DocumentProcessor`
3. The processed documents are indexed in ChromaDB using the `ChromaIndexer`
4. The documents are stored in the `documents` dictionary for later use

This flow is triggered by the `--process_docs` command-line argument:

```python
if args.process_docs:
    # Check if sources are provided
    if not args.sources:
        logger.error("No sources provided. Use --sources to specify file paths or URLs.")
        sys.exit(1)
        
    # Convert sources to the expected format
    formatted_sources = []
    for source in args.sources:
        # Determine if source is a URL or a file
        if source.startswith(('http://', 'https://')):
            formatted_sources.append({"source": source, "type": "url"})
        else:
            formatted_sources.append({"source": source, "type": "pdf"})
    
    # Process documents
    system.process_documents(formatted_sources)
    print(f"Successfully processed {len(formatted_sources)} documents.")
```

## Part 2: Q/A and Summary with Agent

The Q/A and summary functionality is implemented using a LangGraph workflow. LangGraph is a framework for building stateful, multi-step AI applications using a graph-based approach. Let's look at the key components:

### RAGState TypedDict

The `RAGState` TypedDict defines the state that is passed between nodes in the LangGraph workflow:

```python
class RAGState(TypedDict):
    query: str
    query_type: Optional[str]
    documents: Optional[List[Document]]
    document_summaries: Optional[List[Dict]]
    generation: Optional[str]
    final_answer: Optional[str]
```

### LangGraph Nodes

The LangGraph workflow consists of several nodes, each responsible for a specific task:

1. `classify_query`: Classifies a query as either 'qa' or 'summary':

```python
def classify_query(state: RAGState) -> RAGState:
    """
    Classify a query as either 'qa' or 'summary'
    """
    logger.info(f"Classifying query: {state['query']}")
    
    # Create prompt for classification
    prompt = PromptTemplate(
        template="""You are a query classifier. Your task is to determine if a user query is asking for:
        1. A question-answering task (labeled as 'qa') - where the user is seeking specific information or answers
        2. A summarization task (labeled as 'summary') - where the user is asking for a summary or overview
        
        Respond with ONLY 'qa' or 'summary'.
        
        User query: {query}
        Classification:""",
        input_variables=["query"],
    )
    
    # Create chain
    classification_chain = prompt | llm | StrOutputParser()
    
    # Get classification
    classification = classification_chain.invoke({"query": state["query"]}).strip().lower()
    
    # Ensure we get a valid classification
    if classification not in ['qa', 'summary']:
        # Default to qa if classification is unclear
        classification = 'qa'
        
    logger.info(f"Query classified as: {classification}")
    
    # Update state
    return {**state, "query_type": classification}
```

2. `retrieve_documents`: Retrieves relevant documents for the query:

```python
def retrieve_documents(state: RAGState) -> RAGState:
    """
    Retrieve relevant documents for the query
    """
    logger.info(f"Retrieving documents for query: {state['query']}")
    
    # Initialize ChromaIndexer
    indexer = ChromaIndexer()
    
    # Search for documents
    documents = indexer.search(state["query"])
    
    logger.info(f"Retrieved {len(documents)} documents")
    
    # Update state
    return {**state, "documents": documents}
```

3. `answer_question`: Answers a question using RAG:

```python
def answer_question(state: RAGState) -> RAGState:
    """
    Answer a question using RAG
    """
    logger.info(f"Answering question: {state['query']}")
    
    # Format context
    documents = state["documents"]
    formatted_context = ""
    for i, doc in enumerate(documents):
        formatted_context += f"\n--- Document {i+1} ---\n"
        formatted_context += doc.page_content
        formatted_context += f"\n--- End Document {i+1} ---\n"
    
    # Create prompt for answering
    prompt = PromptTemplate(
        template="""You are a helpful assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't find the answer in the context, just say that you don't have that information.
        
        Context:
        {context}
        
        Question: {query}
        Answer:""",
        input_variables=["context", "query"],
    )
    
    # Create chain
    answer_chain = prompt | llm | StrOutputParser()
    
    # Get answer
    answer = answer_chain.invoke({"context": formatted_context, "query": state["query"]})
    
    # Update state
    return {**state, "final_answer": answer}
```

4. `summarize_documents`: Summarizes each document:

```python
def summarize_documents(state: RAGState) -> RAGState:
    """
    Summarize each document
    """
    logger.info("Summarizing documents")
    
    # Get all documents from the system
    documents = []
    
    # Load previously processed documents from the output directory
    md_files = list(Path(OUTPUT_DIR).glob("*.md"))
    
    document_summaries = []
    
    # Load each markdown file
    for md_file in md_files:
        try:
            # Read the markdown file
            with open(md_file, "r") as f:
                content = f.read()
                
            # Create metadata
            metadata = {
                "file_name": md_file.name,
                "source": "loaded_from_disk",
                "processed_time": md_file.stat().st_mtime
            }
            
            # Create prompt for summarization
            prompt = PromptTemplate(
                template="""You are a document summarizer. Your task is to create a concise yet comprehensive summary of the document.
                Focus on the main points, key findings, and important details.
                Make the summary clear, well-structured, and informative.
                
                Document:
                {document}
                
                Summary:""",
                input_variables=["document"],
            )
            
            # Create chain
            summary_chain = prompt | llm | StrOutputParser()
            
            # Get summary
            summary = summary_chain.invoke({"document": content})
            
            # Add to summaries
            document_summaries.append({
                "summary": summary,
                "metadata": metadata
            })
            
            logger.info(f"Summarized document: {md_file.name}")
            
        except Exception as e:
            logger.error(f"Error summarizing document {md_file}: {str(e)}")
    
    # Update state
    return {**state, "document_summaries": document_summaries}
```

5. `create_final_summary`: Creates a final summary from individual document summaries:

```python
def create_final_summary(state: RAGState) -> RAGState:
    """
    Create a final summary from individual document summaries
    """
    logger.info("Creating final summary")
    
    # Format summaries
    summaries = state["document_summaries"]
    formatted_summaries = ""
    for i, summary in enumerate(summaries):
        formatted_summaries += f"\n--- Document {i+1}: {summary['metadata'].get('file_name', 'unknown')} ---\n"
        formatted_summaries += summary["summary"]
        formatted_summaries += f"\n--- End Document {i+1} ---\n"
    
    # Create prompt for final summary
    prompt = PromptTemplate(
        template="""You are a document synthesizer. Your task is to create a comprehensive final summary that reorganizes and integrates information from multiple document summaries.
        Identify common themes, highlight key points, and present a coherent overview that connects information across all documents.
        Make the final summary well-structured, informative, and easy to understand.
        
        Document summaries:
        {summaries}
        
        Final summary:""",
        input_variables=["summaries"],
    )
    
    # Create chain
    final_summary_chain = prompt | llm | StrOutputParser()
    
    # Get final summary
    final_summary = final_summary_chain.invoke({"summaries": formatted_summaries})
    
    # Update state
    return {**state, "final_answer": final_summary}
```

6. `route_by_query_type`: Routes to the appropriate node based on query type:

```python
def route_by_query_type(state: RAGState) -> str:
    """
    Route to the appropriate node based on query type
    """
    query_type = state["query_type"]
    
    if query_type == "qa":
        return "qa"
    else:
        return "summary"
```

### LangGraph Workflow

The LangGraph workflow is defined as follows:

```python
# Create the LangGraph workflow
workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("classify", classify_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("answer", answer_question)
workflow.add_node("summarize", summarize_documents)
workflow.add_node("final_summary", create_final_summary)

# Set entry point
workflow.set_entry_point("classify")

# Add edges
workflow.add_edge("classify", "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    route_by_query_type,
    {
        "qa": "answer",
        "summary": "summarize"
    }
)
workflow.add_edge("summarize", "final_summary")
workflow.add_edge("answer", END)
workflow.add_edge("final_summary", END)

# Compile the graph
rag_app = workflow.compile()
```

This workflow can be visualized as follows:

```
classify -> retrieve -> [route_by_query_type] -> qa -> answer -> END
                                              -> summary -> summarize -> final_summary -> END
```

### Query Processing Flow

The query processing flow can be summarized as follows:

1. The user provides a query
2. The `DocumentSystem` processes the query using the LangGraph workflow
3. The workflow classifies the query as either 'qa' or 'summary'
4. If the query is classified as 'qa', the workflow retrieves relevant documents and generates an answer
5. If the query is classified as 'summary', the workflow summarizes each document and creates a final summary
6. The final answer is returned to the user

This flow is triggered by the `--query` command-line argument:

```python
elif args.query:
    # Check if documents have been processed
    if not system.documents:
        logger.warning("No documents have been processed. Results may be limited.")
        
    # Process query
    response = system.process_query(args.query)
    print("\nResponse:")
    print(response)
```

The `process_query` method of the `DocumentSystem` class is defined as follows:

```python
def process_query(self, query: str) -> str:
    """
    Process a user query using the LangGraph workflow
    
    Args:
        query: User query
        
    Returns:
        Response to the query
    """
    logger.info(f"Processing query: {query}")
    
    # Initialize state
    initial_state = {
        "query": query,
        "query_type": None,
        "documents": None,
        "document_summaries": None,
        "generation": None,
        "final_answer": None
    }
    
    # Run the workflow
    result = rag_app.invoke(initial_state)
    
    # Return the final answer
    return result["final_answer"]
```

## Conclusion

In this tutorial, we've explored how to build a powerful document processing and RAG system using OpenAI's language models and LangGraph. The system can process PDF files and URLs, convert them to markdown, and use them to answer questions or generate summaries.

The key components of the system are:

1. **Data Ingestion**:
   - `DocumentProcessor`: Converts documents to markdown
   - `ChromaIndexer`: Indexes documents in ChromaDB
   - `DocumentSystem`: Integrates the document processor and indexer

2. **Q/A and Summary with Agent**:
   - LangGraph workflow: Processes user queries
   - Nodes: classify_query, retrieve_documents, answer_question, summarize_documents, create_final_summary
   - Routing: Routes queries to the appropriate node based on query type

This system demonstrates the power of combining document processing, vector databases, and language models to create a flexible and powerful RAG system. By using LangGraph, we can create a structured workflow that can handle different types of queries and provide accurate and informative responses.

To use this system, you can run the following commands:

```bash
# Process documents
python ex30.py --process_docs --sources document1.pdf document2.pdf https://example.com/document3.pdf

# Query the system
python ex30.py --query "What is the main topic discussed in these documents?"
```

The system will process the documents, index them in ChromaDB, and use the LangGraph workflow to generate a response to the query.
