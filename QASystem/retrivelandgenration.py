from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.components.builders import PromptBuilder
import os
from dotenv import load_dotenv
from utils import pinecon_configuration


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_TOKEN = os.getenv("HF_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["HF_API_TOKEN"] = HF_API_TOKEN


prompt_template = """Answer the question based on the provided context.
If the content does not include an answer, reply with only:
"The answer is not in my context" or "Not available in the history."

Query: {{query}}
Documents:
{{documents}}  # Instead of using a for-loop, just inject documents as a single string

Answer:
"""

def get_response(query:str)->str:
    
    query_pipeline = Pipeline()

    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriver", PineconeEmbeddingRetriever(document_store = pinecon_configuration()))
    query_pipeline.add_component("prompt_builder", PromptBuilder(template = prompt_template))
    query_pipeline.add_component("Model", GoogleAIGeminiGenerator(model="gemini-2.0-flash",api_key=Secret.from_token(GEMINI_API_KEY)))

    # Connect components
    query_pipeline.connect("text_embedder.embedding", "retriver.query_embedding")
    query_pipeline.connect("retriver.documents", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder", "Model")

    response = query_pipeline.run(
        {
            "text_embedder" : {"text":query},
            "prompt_builder" : {"query" : query}
        }
    )
    return response



if __name__ == "__main__":
    result = get_response("what is this project about?")
    print(result)
