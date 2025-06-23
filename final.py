from sentence_transformers import SentenceTransformer
import faiss
import pickle
import google.generativeai as genai
import streamlit as st

st.set_page_config(page_title="Legal Assistant")
st.title("RTI Legal Assistant")

GEMINI_API_KEY = "AIzaSyAc_IaJ5dTGKL6VOjpPQK1gX7CjiPiNnrw"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
act_choice = st.selectbox("📚 Choose a Law", ["Right To Information Act,2005", "Code of Civil Procedure,1908"])

@st.cache_resource
def load_resources(act_name):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index_path = f"./Acts/{act_name}/{act_name}.index"
    pkl_path = f"./Acts/{act_name}/{act_name}.pkl"

    index = faiss.read_index(index_path)
    with open(pkl_path, "rb") as f:
        sections = pickle.load(f)

    return model,index, sections
model,index,sections=load_resources(act_choice)

def search_faiss(query, k=7):
    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, k)
    results = [sections[i] for i in I[0]]
    return results


def ask_gemini(query, sections):
    context = "\n\n".join([f"Section {s['section']} - {s['title']}\n{s['text']}" for s in sections])
    prompt = f"""
You are a legal assistant. Based on the following sections of the {act_choice} Act, answer the question clearly and concisely.

Question: {query}

Relevant Section
{context}

Answer:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini: {e}"

if "chat" not in st.session_state:
    st.session_state.chat = []

query=st.chat_input("Ask You Question")
if query:
    st.session_state.chat.append(("user", query))  # Save user message
    with st.spinner("Searching legal knowledge base..."):
        top_sections= search_faiss(query)
        answer = ask_gemini(query, top_sections)
    st.session_state.chat.append(("ai", answer))
    for role, msg in st.session_state.chat:
        if role == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)
