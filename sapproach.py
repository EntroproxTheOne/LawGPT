from sentence_transformers import SentenceTransformer
import faiss
import pickle
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyAc_IaJ5dTGKL6VOjpPQK1gX7CjiPiNnrw"
genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini model
gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

# === Load model, FAISS index, and section metadata ===
print("🔁 Loading model and data...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("rti_faiss.index")

with open("rti_sections.pkl", "rb") as f:
    sections = pickle.load(f)

print("✅ Model and index loaded.")
# === Semantic Search ===
def search_faiss(query, k=7):
    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, k)
    results = [sections[i] for i in I[0]]
    return results


def ask_gemini(query, sections):
    context = "\n\n".join([f"Section {s['section']} - {s['title']}\n{s['text']}" for s in sections])
    prompt = f"""
You are a legal assistant. Based on the following sections of the RTI Act, answer the question clearly and concisely.

Question: {query}

Relevant Sections:
{context}

Answer:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini: {e}"

# === Main Query Loop ===
while True:
    query = input("\n❓ Ask your legal question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    results = search_faiss(query)

    print("\n🔎 Top matching sections:\n")
    for i, sec in enumerate(results):
        print(f"{i + 1}. 📘 {sec['section']} - {sec['title']}")
        print(f"📝 {sec['text']}\n")
    answer = ask_gemini(query,results)
    print("\n🧠 Gemini's Answer:\n")
    print(answer)