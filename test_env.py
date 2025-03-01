from dotenv import load_dotenv
import os
from datasets import load_dataset
from huggingface_hub import HfApi

load_dotenv()

print("Environment Validation Check:")
print("Current directory:", os.getcwd())
print("Files present:", os.listdir())

print("\nAPI Key Validation:")
print("Cohere key exists:", bool(os.getenv("COHERE_API_KEY")))
print("Voyage key exists:", bool(os.getenv("VOYAGE_API_KEY")))
print("OpenAI key exists:", bool(os.getenv("OPENAI_API_KEY")))

print("\nDataset Availability Check:")
try:
    dataset = load_dataset("PatronusAI/financebench", split="train")
    print("Dataset loaded successfully")
    print(f"Sample structure: {dataset[0].keys()}")
except Exception as e:
    print(f"Dataset loading failed: {str(e)}")

api = HfApi()
dataset_info = next(iter(api.list_datasets(search="PatronusAI/financebench")), None)
print("Dataset exists on Hub:", "Yes" if dataset_info else "No")

print("\nEmbedding Dimension Check:")
try:
    from main import CohereEmbeddingsWrapper, VoyageEmbeddings
    cohere_emb = CohereEmbeddingsWrapper()
    voyage_emb = VoyageEmbeddings()
    print(f"Cohere embedding dim: {len(cohere_emb.embed_query('test'))}")
    print(f"Voyage embedding dim: {len(voyage_emb.embed_query('test'))}")
except Exception as e:
    print(f"Embedding test failed: {str(e)}")