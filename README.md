# 📄 Chat with PDF - Hybrid Search using Haystack & Streamlit

## 🚀 Project Overview
This project enables users to **upload PDFs** and **ask questions** about their content using a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline. It integrates:

- **SentenceTransformersTextEmbedder** for text embedding
- **PineconeEmbeddingRetriever** for vector-based document retrieval
- **GoogleAIGeminiGenerator** for AI-powered responses
- **Streamlit UI** for user interaction

## 🛠 Tech Stack
- **Python** (FastAPI + Streamlit)
- **Haystack** (NLP pipeline)
- **Pinecone** (Vector database for document storage & retrieval)
- **Google AI Gemini** (LLM-based response generation)
- **Jinja2** (Prompt templating)

---

## 📌 Features
✅ Upload a **PDF document** and ask questions about it.
✅ **Hybrid Retrieval**: Uses **vector search** and **semantic similarity**.
✅ **Dynamic Jinja2 Prompts** for better LLM responses.
✅ **Session-based Cleanup**: Deletes user data after session ends.
✅ **Error Handling & Debugging**: Logs retrieval steps for easy troubleshooting.

---

## 📂 Project Structure
```
📁 QASystem
 ┣ 📂 utils
 ┃ ┗ 📄 pinecone_configuration.py  # Pinecone DB setup
 ┣ 📄 app.py  # Main Streamlit app
 ┣ 📄 requirements.txt  # Dependencies
 ┣ 📄 .env  # API Keys & Secrets
 ┗ 📄 README.md  # Documentation
```

---

## 🏗 Installation & Setup
### 1️⃣ Clone Repository
```bash
git clone https://github.com/Warishayat/Hybrid-Rag-Search-Haystack
cd chat-with-pdf
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirement.txt
```

### 4️⃣ Set Up Environment Variables
Create a **.env** file and add:
```ini
PINECONE_API_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_google_ai_gemini_api_key
HF_API_TOKEN=your_huggingface_api_token
```

---

## 🚀 Run the Application
```bash
streamlit run main.py
```

---

## 🛠 API Flow
1. **Upload PDF** → Extracts text & stores embeddings in **Pinecone**.
2. **User Query** → Gets transformed into an **embedding**.
3. **Retriever** → Searches for similar document chunks.
4. **Jinja2 Prompt Builder** → Structures the query context.
5. **Google AI Gemini** → Generates the final response.
6. **Streamlit UI** → Displays the answer.

---


## 🛠 Troubleshooting
- **"No response from the model"** → Check if API keys are correct.
- **"PDF uploaded successfully but no result"** → Ensure embeddings are being stored in Pinecone.
- **Debugging Retrieval** → Add `print(retrieved_docs)` in `get_response()` to check document retrieval.

---

## 📝 Future Improvements
🔹 Add support for **multiple PDFs** in a single query.  
🔹 Improve **response ranking** for better relevance.  
🔹 Integrate **local LLMs** (Mistral, Llama 3) instead of cloud-based AI.  

---

## 🤝 Contributing
Pull requests are welcome! Feel free to improve the code and submit changes.

---
