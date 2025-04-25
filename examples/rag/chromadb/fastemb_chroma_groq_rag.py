from fastembed import TextEmbedding
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import numpy as np
import os
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# Load environment variables
load_dotenv(override=True)

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en"  # FastEmbed model
COLLECTION_NAME = "fastemb_rag_collection"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

# Initialize FastEmbed embedding model
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP
)

class FastEmbedChromaGroqRAG:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"Created new collection: {self.collection_name}")
        
        # Initialize LLM for generation (Groq)
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",  # You can change to another Groq model
            temperature=0.2,
        )
    
    def _embed_texts(self, texts: List[str], is_query: bool = False) -> List[np.ndarray]:
        """Generate embeddings for texts using FastEmbed"""
        if is_query:
            # Use query_embed for queries
            embeddings = list(self.embedding_model.query_embed(texts))
        else:
            # Use passage_embed for documents
            embeddings = list(self.embedding_model.passage_embed(texts))
        return embeddings
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store"""
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Generate IDs for chunks
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Extract text and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        embeddings = self._embed_texts(texts)
        
        # Convert embeddings to list format for ChromaDB
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        # Convert metadata to string values for ChromaDB compatibility
        processed_metadatas = []
        for metadata in metadatas:
            processed_metadata = {}
            for k, v in metadata.items():
                processed_metadata[k] = str(v)
            processed_metadatas.append(processed_metadata)
        
        # Add data to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=processed_metadatas
        )
        
        print(f"Added {len(chunks)} chunks to collection {self.collection_name}")
        return ids
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self._embed_texts([query], is_query=True)[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # Convert results to Documents
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append(
                Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
            )
        
        return documents
    
    def query(self, query: str, k: int = 3) -> str:
        """Perform RAG query: retrieve documents and generate answer"""
        # Retrieve relevant documents
        docs = self.similarity_search(query, k=k)
        
        if not docs:
            return "No relevant documents found to answer your query."
        
        # Format documents for context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer using Groq LLM
        prompt = f"""You are an assistant that answers questions based on the provided context.
        
Context:
{context}

Question: {query}

Answer the question based on the provided context. If the context doesn't contain relevant information to answer the question, clearly state that you don't have that information."""
        
        response = self.llm.invoke(prompt)
        return response.content

def main():
    # Create RAG system
    rag = FastEmbedChromaGroqRAG()
    
    # Sample documents (replace with your own documents)
    sample_docs = [
        Document(
            page_content="Maharana Pratap was a Rajput warrior king from Mewar. He fought against the Mughal Empire led by Akbar.",
            metadata={"source": "history.txt"}
        ),
        Document(
            page_content="The Battle of Haldighati in 1576 was his most famous battle. He refused to submit to Akbar and continued guerrilla warfare.",
            metadata={"source": "battles.txt"}
        ),
        Document(
            page_content="His capital was Chittorgarh, which he lost to the Mughals. He died in 1597 at the age of 57.",
            metadata={"source": "biography.txt"}
        ),
        Document(
            page_content="Maharana Pratap is considered a symbol of Rajput resistance against foreign rule. His legacy is celebrated in Rajasthan through festivals and monuments.",
            metadata={"source": "legacy.txt"}
        ),
    ]
    
    # Add documents to RAG system
    rag.add_documents(sample_docs)
    
    # Example query
    query = "Who was Maharana Pratap and what is he known for?"
    print(f"\nQuery: {query}")
    
    # Get response
    response = rag.query(query)
    print(f"\nResponse: {response}")

if __name__ == "__main__":
    main() 