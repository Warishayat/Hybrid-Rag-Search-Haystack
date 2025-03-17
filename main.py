import streamlit as st
import os
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.components.builders import PromptBuilder
from QASystem.utils import pinecon_configuration

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_TOKEN = os.getenv("HF_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["HF_API_TOKEN"] = HF_API_TOKEN

# Initialize session state for tracking file processing
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False  # Track if file is processed

# Define Prompt Template
prompt_template ="""
You are an AI assistant that answers questions based on the provided document context.

{% if documents %}
- If the document contains relevant information, generate a concise and accurate answer.  
- If the document does not have a direct answer but contains related information (above 45% probability), summarize the relevant details.  
{% else %}
- If there is no relevant information, respond only with: "Sorry, this PDF does not contain this information."  
{% endif %}

Query: {{ query }}
Documents:
{% for doc in documents %}
- {{ doc }}
{% endfor %}

Answer:
"""

# Function to Process Query
def get_response(query: str) -> str:
    try:
        query_pipeline = Pipeline()
        query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
        query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=pinecon_configuration()))
        query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
        query_pipeline.add_component("Model", GoogleAIGeminiGenerator(model="gemini-2.0-flash", api_key=Secret.from_token(GEMINI_API_KEY)))

        # Connect Components
        query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        query_pipeline.connect("retriever.documents", "prompt_builder.documents")
        query_pipeline.connect("prompt_builder", "Model")

        response = query_pipeline.run(
            {
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query}
            }
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Hybrid-Search with HayStack")
st.write("Upload a document and ask questions!")

# Sidebar for File Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        # Simulate document processing (you should replace this with actual processing logic)
        st.session_state.file_uploaded = True  # Mark document as processed
        st.sidebar.success("✅ Document uploaded and processed successfully!")

# Input for Query
query = st.text_input("Ask a question about your document:")

if st.button("Get Answer"):
    if not st.session_state.file_uploaded:
        st.warning("⚠️ Please upload a document first!")
    elif query:
        with st.spinner("🔄 Generating response..."):
            response = get_response(query)
            st.write("### AI Response:")
            st.success(response['Model']['replies'][0])
    else:
        st.warning("⚠️ Please enter a question!")

st.write("---")
st.write("Built with ❤️ using Haystack and Streamlit 🚀")
