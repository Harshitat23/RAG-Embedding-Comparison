# RAG-Embedding-Comparison
This project evaluates different embedding models (Cohere and Voyage AI) in a Retrieval-Augmented Generation (RAG) system. The system retrieves relevant context using FAISS and generates answers using GPT-4. The evaluation uses finance-related questions from the **PatronusAI/financebench** dataset.

## 🔍 Key Features
Custom Secure Embedding Implementations:

**-> Voyage AI Embeddings (via voyageai.Client)**
**-> Cohere Embeddings (via cohere.Client)**
Vector Database for Retrieval: Uses FAISS for fast and efficient similarity search.
GPT-4 for Answer Generation: Generates answers based on retrieved financial context.
Quantitative Evaluation Metrics:

-> Exact Match Score: Checks if the generated response matches the reference answer.
-> BERT Score: Evaluates semantic similarity of generated responses.
-> Retrieval Precision: Measures how accurately retrieved documents contain relevant information.

## 🛠️ Setup & Installation

**1) Install Dependencies**
pip install -r requirements.txt

**2) Set Up API Keys**
Create a .env file and add the required API keys:
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
VOYAGE_API_KEY=your_voyageai_api_key

**3) Run the Evaluation**
Execute the script to compare Cohere and Voyage AI embeddings - 
python main.py




**Name: Harshita
Email: harshitat010@gmail.com**
