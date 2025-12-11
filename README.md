# Agentic RAG Medical Chatbot (Qdrant + Chainlit)

This project is an **Agentic Retrieval-Augmented Generation (RAG)** chatbot designed to answer medical questions using curated datasets and intelligent retrieval. It combines **Qdrant vector search**, **LangGraph agent workflows**, **Sentence Transformer embeddings**, **Chainlit UI**, and **OpenAI models** to deliver concise, context-aware responses.

This chatbot is deployed on **Render**, using **environment variables for security**, and retrieves information from vectorized Kaggle medical datasets stored in **Qdrant Cloud**.
 

---

## 🌟 What the App Does

- Accepts a user’s medical or device-related question  
- Routes the question to the appropriate data source using an **agentic router**  
- Retrieves relevant context using **semantic search via Qdrant**  
- Checks whether the retrieved context is relevant  
- Falls back to **web search** (Tavily) if needed  
- Builds an optimized RAG prompt  
- Generates a short, reliable answer using an **OpenAI LLM**

The chatbot uses real medical datasets (manuals + Q&A) to answer questions with high-quality, dataset-grounded context.

---

## 🧠 How the System Works

### 1️⃣ Agentic Router (LangGraph)
Determines the best information source based on the user query:

- **Medical Q&A dataset**
- **Medical Device Manuals dataset**
- **Web search** (if the query is outside dataset scope)

### 2️⃣ Vector Retrieval (Qdrant Cloud)
All dataset entries are converted into embeddings using a **Sentence Transformer** model and stored in Qdrant.

For each query:

- The question is embedded  
- Qdrant returns the most similar documents  
- These documents become the RAG context  

### 3️⃣ Context Relevance Check
An LLM checks:

> “Is this retrieved context actually relevant to the question?”

If **not relevant**, the workflow retries or uses web search.

### 4️⃣ Prompt Construction
A final prompt is built using:

- Retrieved context  
- User question  
- Safety and brevity instructions  

### 5️⃣ Answer Generation
An OpenAI model generates a ~50-word answer based on combined context and prompt instructions.

---

## 🛠️ Technologies Used

### **🔹 Qdrant (Vector Database)**
- Stores embeddings for both Kaggle datasets  
- Provides high-speed semantic search with `query_points`

### **🔹 Sentence Transformers**
- Model: `all-MiniLM-L6-v2`  
- Converts text into dense embeddings

### **🔹 Kaggle Datasets**
Two curated public datasets:

- Global Medical Device Manuals  
- Comprehensive Medical Q&A dataset  

Downloaded through Kaggle API → processed → embedded → uploaded to Qdrant.

### **🔹 LangGraph (Agent Workflow Engine)**
Implements:

- Router agent  
- Relevance agent  
- Prompt builder agent  
- LLM generator agent  

### **🔹 Chainlit**
- Provides a clean, interactive web-based chat UI  
- Handles real-time conversation with the RAG pipeline

### **🔹 OpenAI API**
- Used for routing, relevance checking, and final answer generation

### **🔹 Render (Hosting)**
- Hosts the Chainlit application  
- Environment variables store API keys securely

---

 
