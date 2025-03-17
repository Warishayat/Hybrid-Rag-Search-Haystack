#setup the database
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from dotenv import load_dotenv
import os
import pinecone


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HF_API_TOKEN"] = HF_TOKEN


#load the database
def pinecon_configuration():
    document_store = PineconeDocumentStore(
	index="haystack",
	namespace="haystack",
	dimension=768,
  	metric="cosine",
  	spec={"serverless": {"region": "us-east-1", "cloud": "aws"}}
    )

    return document_store

    print("Sucessfully initialized")

if __name__ == '__main__':
    print("This is working fine")