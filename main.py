import os
import cohere
import voyageai
import numpy as np
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import wandb
from bert_score import BERTScorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()

class SecureEmbeddings:
    """Base class for secure embedding implementations"""
    def __init__(self, api_key_name):
        self.api_key = os.getenv(api_key_name)
        self.validate_key()

    def validate_key(self):
        if not self.api_key or len(self.api_key) < 20:
            raise ValueError("Invalid API key detected")

class VoyageEmbeddings(SecureEmbeddings, Embeddings):
    """Voyage AI embedding implementation"""
    def __init__(self):
        super().__init__("VOYAGE_API_KEY")
        voyageai.api_key = self.api_key
    
    def embed_documents(self, texts):
        try:
            result = voyageai.embed(texts, model="voyage-01")
            return result.embeddings
        except Exception as e:
            print(f"Voyage API Error: {str(e)}")
            return []
    
    def embed_query(self, text):
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else [0.0]*1024

class CohereEmbeddingsWrapper(SecureEmbeddings, Embeddings):
    """Cohere embedding implementation"""
    def __init__(self):
        super().__init__("COHERE_API_KEY")
        self.client = cohere.Client(self.api_key)
    
    def embed_documents(self, texts):
        try:
            response = self.client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings
        except Exception as e:
            print(f"Cohere API Error: {str(e)}")
            return []
    
    def embed_query(self, text):
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else [0.0]*1024

class RAGEvaluator:
    def __init__(self, embedding_model, model_name):
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len
        )
        self.vector_store = None
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def _create_vector_store(self, documents):
        texts = self._process_documents(documents)
        if not texts:
            raise ValueError("No valid documents processed for vector store creation")
        return FAISS.from_texts(texts, self.embedding_model)

    def _process_documents(self, documents):
        processed_texts = []
        for doc in documents:
            if "evidence" not in doc or "answer" not in doc:
                continue
            
            try:
                evidence = doc["evidence"]
                if isinstance(evidence, list):
                    evidence_text = " ".join(
                        str(item.get("text", item)) 
                        for item in evidence
                        if item is not None
                    )
                else:
                    evidence_text = str(evidence)
                
                if not evidence_text.strip():
                    continue
                    
                chunks = self.text_splitter.split_text(evidence_text)
                processed_texts.extend(chunks)
            except Exception as e:
                print(f"Error processing document: {str(e)}")
                continue
        return processed_texts

    def _retrieve_context(self, question, k=3):
        try:
            query_embedding = self.embedding_model.embed_query(question)
            docs = self.vector_store.similarity_search_by_vector(query_embedding, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return []

    def generate_answer(self, question):
        try:
            context = self._retrieve_context(question)
            prompt = ChatPromptTemplate.from_template(
                """Answer the question based only on the following context:
                {context}
                
                Question: {question}
                """
            )
            chain = prompt | self.llm
            return chain.invoke({"context": context, "question": question}).content
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return "Error generating answer"

class EvaluationFramework:
    def __init__(self):
        wandb.init(project="rag-evaluation")
        self.bertscorer = BERTScorer(lang="en", model_type="roberta-large")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_retrieval_precision(self, reference_answer, retrieved_docs):
        try:
            reference_embedding = self.similarity_model.encode([reference_answer])
            doc_embeddings = self.similarity_model.encode(retrieved_docs)
            similarities = cosine_similarity(reference_embedding, doc_embeddings)
            return float(np.mean(similarities))
        except Exception as e:
            print(f"Precision calculation error: {str(e)}")
            return 0.0

    def evaluate_answer(self, question, reference, prediction, retrieved_docs):
        try:
            exact_match = int(prediction.strip().lower() == reference["answer"].strip().lower())
            bert_score = self.bertscorer.score([prediction], [reference["answer"]])[2].mean().item()
            retrieval_precision = self.calculate_retrieval_precision(reference["answer"], retrieved_docs)
            
            wandb.log({
                "exact_match": exact_match,
                "bert_score": bert_score,
                "retrieval_precision": retrieval_precision,
                "question": question,
                "prediction": prediction,
                "reference": reference["answer"]
            })
            
            return {
                "exact_match": exact_match,
                "bert_score": bert_score,
                "retrieval_precision": retrieval_precision,
            }
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return {
                "exact_match": 0,
                "bert_score": 0.0,
                "retrieval_precision": 0.0,
            }

def main():
    try:
        evaluator = EvaluationFramework()
        dataset = load_dataset("PatronusAI/financebench", split="train")
        test_samples = dataset.select(range(10))
        
        print("\nSample document structure:")
        print(test_samples[0].keys())
        print("Evidence type:", type(test_samples[0]["evidence"]))
        
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return

    cohere_emb = CohereEmbeddingsWrapper()
    voyage_emb = VoyageEmbeddings()

    results = {}
    for model in [(cohere_emb, "Cohere"), (voyage_emb, "Voyage")]:
        try:
            print(f"\nEvaluating {model[1]} model...")
            rag = RAGEvaluator(model[0], model[1])
            rag.vector_store = rag._create_vector_store(test_samples)
            
            model_results = []
            for idx, sample in enumerate(test_samples):
                try:
                    print(f"Processing sample {idx+1}/{len(test_samples)}")
                    answer = rag.generate_answer(sample["question"])
                    retrieved_docs = rag._retrieve_context(sample["question"])
                    metrics = evaluator.evaluate_answer(
                        sample["question"],
                        sample,
                        answer,
                        retrieved_docs
                    )
                    model_results.append(metrics)
                except Exception as e:
                    print(f"Error processing sample {idx}: {str(e)}")
                    model_results.append({
                        "exact_match": 0,
                        "bert_score": 0.0,
                        "retrieval_precision": 0.0,
                    })
            
            results[model[1]] = model_results
        except Exception as e:
            print(f"Failed to evaluate {model[1]}: {str(e)}")
            continue

    analysis = {}
    for model_name, metrics in results.items():
        analysis[model_name] = {
            "exact_match": np.mean([m["exact_match"] for m in metrics]),
            "bert_score": np.mean([m["bert_score"] for m in metrics]),
            "retrieval_precision": np.mean([m["retrieval_precision"] for m in metrics]),
        }

    print("\nFinal Comparative Analysis:")
    for metric in ["exact_match", "bert_score", "retrieval_precision"]:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Cohere: {analysis.get('Cohere', {}).get(metric, 0):.3f}")
        print(f"  Voyage: {analysis.get('Voyage', {}).get(metric, 0):.3f}")

if __name__ == "__main__":
    main()