import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# === STEP 1: Load legal data from JSON ===
with open("Right To Information Act,2025.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [f"{item['section']} - {item['title']}: {item['text']}" for item in data]

# === STEP 2: Load sentence-transformer embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")  # Or use other models like all-mpnet-base-v2

# === STEP 3: Generate embeddings ===
print("🔄 Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# Ensure float32 for FAISS
embeddings = np.array(embeddings).astype("float32")

# === STEP 4: Create and populate FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === STEP 5: Save the FAISS index and section metadata ===
faiss.write_index(index, "rti_faiss.index")

with open("rti_sections.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Embeddings generated and saved to 'rti_faiss.index'")
print("✅ Metadata saved to 'rti_sections.pkl'")
