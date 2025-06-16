from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from collections import defaultdict
import google.generativeai as genai


# Replace this with your actual Gemini API key
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

# === Build mappings for cross-reference support ===
section_references = defaultdict(list)
section_number_to_data = {sec['section']: sec for sec in sections}

for sec in sections:
    referenced = set(re.findall(r'\bsection\s+(\d+)', sec['text'].lower()))
    if referenced:
        section_references[sec['section']] = [
            int(r) for r in referenced if int(r) != sec['section']
        ]

# === INTENT SECTION MAP ===
INTENT_SECTION_MAP = {
    'definition': [2, 3],
    'application': [6],
    'timing': [7, 11, 19],
    'third_party': [11],
    'exemptions': [8, 9, 10],
    'appeals': [18, 19],
    'penalties': [20],
    'obligations': [4, 5],
    'fees': [6, 7],
    'monitoring': [25, 26],
    'rule_making': [27, 28],
    'information_commissions': [12, 13, 14, 15, 16, 17],
    'jurisdiction': [23],
    'override': [22],
    'intelligence_exemption': [21, 24],
    'repeal_savings': [29, 30, 31],
}

# === INTENT KEYWORDS ===
INTENT_KEYWORDS = {
    'definition': ['what is rti', 'define', 'meaning of rti'],
    'application': ['how to apply', 'file rti', 'submit request', 'application process'],
    'timing': ['how long', 'duration', 'days', 'delay', 'time limit'],
    'third_party': ['third party', 'personal info', 'private data', 'someone else'],
    'exemptions': ['exempted', 'denied', 'not disclosed', 'withheld', 'refused'],
    'appeals': ['appeal', 'complaint', 'rejected', 'not responded', 'unsatisfied'],
    'penalties': ['penalty', 'fine', 'punishment', 'non compliance'],
    'obligations': ['duties', 'responsibility', 'public authority must'],
    'fees': ['fee', 'payment', 'cost', 'charges'],
    'monitoring': ['annual report', 'monitoring', 'supervision', 'review'],
    'rule_making': ['rules', 'regulations', 'framed', 'amendment'],
    'information_commissions': ['cic', 'sic', 'central information', 'state commission'],
    'jurisdiction': ['can court hear', 'barred', 'jurisdiction', 'go to court'],
    'override': ['supersede', 'override other laws', 'prevail over'],
    'intelligence_exemption': ['intelligence agencies', 'security org', 'not apply to', 'raw', 'narcotics'],
    'repeal_savings': ['repeal', 'savings clause', 'earlier law', 'continuation'],
}

# === Intent Classifier ===
def classify_intent(query):
    query_lower = query.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return intent
    return 'generic'

# === Semantic Search ===
def search_faiss(query, k=5):
    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, k)
    results = [sections[i] for i in I[0]]
    return results, D[0]

# === Fallback Search ===
def keyword_fallback(query, data, max_results=3):
    keywords = query.lower().split()
    matches = []
    for section in data:
        if any(kw in section["text"].lower() or kw in section["title"].lower() for kw in keywords):
            matches.append(section)
            if len(matches) >= max_results:
                break
    return matches

# === Include Cross-References ===
def include_cross_references(results, max_refs=3):
    seen_sections = set(sec['section'] for sec in results)
    extended_results = results.copy()

    for sec in results:
        refs = section_references.get(sec['section'], [])
        for ref in refs:
            if ref not in seen_sections and ref in section_number_to_data:
                extended_results.append(section_number_to_data[ref])
                seen_sections.add(ref)
            if len(extended_results) >= len(results) + max_refs:
                break

    return extended_results

# === Optional: Intent-aware reranking ===
def rerank_by_intent(intent, retrieved_sections):
    if intent not in INTENT_SECTION_MAP or intent == 'generic':
        return retrieved_sections  # No reranking

    target_sections = INTENT_SECTION_MAP[intent]
    primary = [sec for sec in retrieved_sections if sec['section'] in target_sections]
    secondary = [sec for sec in retrieved_sections if sec['section'] not in target_sections]
    return primary + secondary

# === Final Search Logic ===
def search_with_fallback(query, k=5, threshold=1.5):
    intent = classify_intent(query)
    faiss_results, scores = search_faiss(query, k)

    if len(faiss_results) == 0 or min(scores) > threshold:
        print("⚠️ FAISS results weak or irrelevant. Using fallback search.")
        results = keyword_fallback(query, sections)
    else:
        results = rerank_by_intent(intent, faiss_results)

    results = include_cross_references(results)
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

    results = search_with_fallback(query)

    print("\n🔎 Top matching sections:\n")
    for i, sec in enumerate(results):
        print(f"{i + 1}. 📘 Section {sec['section']} - {sec['title']}")
        print(f"📝 {sec['text']}\n")
    answer = ask_gemini(query,results)
    print("\n🧠 Gemini's Answer:\n")
    print(answer)