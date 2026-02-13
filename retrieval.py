try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

try:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError:
    from langchain.embeddings import SentenceTransformerEmbeddings

try:
    from langchain.docstore.document import Document
except ImportError:
    # Create a simple Document class if not available
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # Create a simple text splitter if not available
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", ". ", " "]
        
        def split_text(self, text):
            # Simple chunking implementation
            chunks = []
            words = text.split()
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk = " ".join(words[i:i + self.chunk_size])
                chunks.append(chunk)
            return chunks

import os
import json
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriculturalKnowledgeBase:
    """RAG system for agricultural knowledge retrieval"""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base/agri_knowledge.json"):
        self.knowledge_base_path = knowledge_base_path
        self.embeddings = None
        self.vector_store = None
        self.documents = []
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings and vector store"""
        try:
            # Initialize embeddings model (multilingual)
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            logger.info("Embeddings model loaded")
            
            # Load knowledge base and create vector store
            self._load_knowledge_base()
            self._create_vector_store()
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {str(e)}")
    
    def _load_knowledge_base(self):
        """Load knowledge from JSON file"""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            # Convert to Document objects
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            for item in knowledge_data:
                content = f"Topic: {item.get('topic', '')}\nContent: {item.get('content', '')}\nLanguage: {item.get('lang', 'en')}"
                
                # Split into chunks for better retrieval
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "topic": item.get('topic', ''),
                            "language": item.get('lang', 'en'),
                            "id": item.get('id', 0)
                        }
                    )
                    self.documents.append(doc)
            
            logger.info(f"Loaded {len(self.documents)} document chunks")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
    
    def _create_vector_store(self):
        """Create FAISS vector store from documents"""
        try:
            if self.documents:
                self.vector_store = FAISS.from_documents(
                    self.documents, 
                    self.embeddings
                )
                logger.info("Vector store created successfully")
            else:
                logger.warning("No documents to create vector store")
                
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
    
    def retrieve_context(self, query: str, language: str = "en", k: int = 3) -> str:
        """
        Retrieve relevant context for a query
        
        Args:
            query (str): User query
            language (str): Query language
            k (int): Number of documents to retrieve
            
        Returns:
            str: Retrieved context as formatted text
        """
        try:
            if not self.vector_store:
                return "No knowledge base available. Please ensure the knowledge base is properly initialized."
            
            # Search for similar documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            if not docs:
                return "No specific information found for your query."
            
            # Format context
            context_parts = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                context_parts.append(
                    f"Source {i} ({metadata.get('language', 'unknown')}):\n{doc.page_content}"
                )
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return "Error retrieving information. Please try again."

# Global instance
_knowledge_base = None

def get_knowledge_base():
    """Get or create knowledge base instance"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = AgriculturalKnowledgeBase()
    return _knowledge_base

def retrieve_context(query: str, language: str = "en", k: int = 3) -> str:
    """
    Retrieve relevant agricultural context for a query
    
    Args:
        query (str): User query
        language (str): Query language for filtering
        k (int): Number of results to return
        
    Returns:
        str: Formatted context from knowledge base
    """
    kb = get_knowledge_base()
    return kb.retrieve_context(query, language, k)

def add_knowledge_entry(topic: str, content: str, language: str, entry_id: int = None):
    """
    Add new knowledge entry to the knowledge base
    
    Args:
        topic (str): Topic title
        content (str): Content text
        language (str): Language code
        entry_id (int): Optional ID for the entry
    """
    try:
        # Load existing knowledge
        with open("knowledge_base/agri_knowledge.json", 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
        
        # Create new entry
        new_entry = {
            "id": entry_id or (max([item.get('id', 0) for item in knowledge_data]) + 1),
            "topic": topic,
            "content": content,
            "lang": language
        }
        
        knowledge_data.append(new_entry)
        
        # Save back to file
        with open("knowledge_base/agri_knowledge.json", 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Added new knowledge entry: {topic}")
        
        # Reinitialize knowledge base
        global _knowledge_base
        _knowledge_base = None
        
    except Exception as e:
        logger.error(f"Failed to add knowledge entry: {str(e)}")
