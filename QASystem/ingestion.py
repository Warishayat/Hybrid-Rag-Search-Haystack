# Ingest the data and store  
from pathlib import Path
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from utils import pinecon_configuration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_API_KEY = os.getenv("HF_TOKEN")

if not PINECONE_API_KEY or not HF_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY or HF_API_TOKEN in environment variables.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HF_API_TOKEN"] = HF_API_KEY

PDF_PATH = Path(r"C:\Users\HP\Desktop\Hybrid-Rag-Search\data\Proposal for Multimodal Retrieval.pdf")

if not PDF_PATH.exists():
    raise FileNotFoundError(f"ERROR: The file '{PDF_PATH}' does not exist.")

def ingest_data(document_store):
    indexing = Pipeline()
    indexing.add_component("DocumentLoader", PyPDFToDocument())
    indexing.add_component("Splitter", DocumentSplitter(split_by="sentence", split_length=2))
    indexing.add_component("Embeddings", SentenceTransformersDocumentEmbedder())
    indexing.add_component("Writer", DocumentWriter(document_store))

    # Connect components
    indexing.connect("DocumentLoader", "Splitter")
    indexing.connect("Splitter", "Embeddings")
    indexing.connect("Embeddings", "Writer")

    # Run the pipeline
    indexing.run({"DocumentLoader": {"sources": [str(PDF_PATH)]}})

if __name__ == "__main__":
    document_store = pinecon_configuration()
    ingest_data(document_store)
    
