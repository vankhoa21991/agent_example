from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import numpy as np
import os
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# Load environment variables
load_dotenv(override=True)

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en"  # FastEmbed model
EMBEDDING_DIM = 384  # Dimension for bge-small-en
COLLECTION_NAME = "fastemb_rag_collection"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Check for Qdrant configurations (local or cloud)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_storage")

# Initialize FastEmbed embedding model
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP
)

class FastEmbedQdrantRAG:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize Qdrant client (cloud or local)
        if QDRANT_URL and QDRANT_API_KEY:
            print(f"Connecting to Qdrant Cloud at {QDRANT_URL}")
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            print(f"Using local Qdrant at {QDRANT_LOCAL_PATH}")
            self.client = QdrantClient(path=QDRANT_LOCAL_PATH)
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
        
        # Initialize LLM for generation
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    
    def _create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        collections = self.client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            print(f"Created new collection: {self.collection_name}")
    
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
        
        # Add data to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                {
                    "id": id,
                    "vector": embedding.tolist(),
                    "payload": {"text": text, "metadata": metadata}
                }
                for id, embedding, text, metadata in zip(ids, embeddings, texts, metadatas)
            ]
        )
        
        print(f"Added {len(chunks)} chunks to collection {self.collection_name}")
        return ids
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self._embed_texts([query], is_query=True)[0]
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k
        )
        
        # Convert results to Documents
        documents = [
            Document(
                page_content=result.payload["text"],
                metadata=result.payload["metadata"]
            )
            for result in results
        ]
        
        return documents
    
    def query(self, query: str, k: int = 3) -> str:
        """Perform RAG query: retrieve documents and generate answer"""
        # Retrieve relevant documents
        docs = self.similarity_search(query, k=k)
        
        if not docs:
            return "No relevant documents found to answer your query."
        
        # Format documents for context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer using LLM
        prompt = f"""You are an assistant that answers questions based on the provided context.
        
Context:
{context}

Question: {query}

Answer the question based on the provided context. If the context doesn't contain relevant information to answer the question, clearly state that you don't have that information."""
        
        response = self.llm.invoke(prompt)
        return response.content

def main():
    # Create RAG system
    rag = FastEmbedQdrantRAG()
    
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