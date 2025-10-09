from sentence_transformers import SentenceTransformer
from together import Together
import os
import faiss
import pickle
import streamlit as st
import time
st.set_page_config(page_title="Legal Assistant")
st.title("Legal Assistant")

st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117;
        color: gray;
        text-align: center;
        padding: 10px;
        font-size: 0.9rem;
        z-index: 100;
        border-top: 1px solid #333;
    }
    .footer a {
        color: #3399ff;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)
footer_container=st.container()
TOGETHER_API_KEY = st.secrets["together_api_key"]  # store this in Streamlit secrets
together = Together(api_key=TOGETHER_API_KEY)
act_choice = st.selectbox("📚 Choose a Law", ["Right To Information Act,2005", "Code of Civil Procedure,1908","Consumer Protection Act,2019"])


@st.cache_resource
def load_resources(act_name):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index_path = f"./Acts/{act_name}/{act_name}.index"
    pkl_path = f"./Acts/{act_name}/{act_name}.pkl"

    index = faiss.read_index(index_path)
    with open(pkl_path, "rb") as f:
        sections = pickle.load(f)

    return model, index, sections


model, index, sections = load_resources(act_choice)


def search_faiss(query, k=7):
    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, k)
    results = [sections[i] for i in I[0]]
    return results


def ask_llama(query, sections):
    context = "\n\n".join([f"Section {s['section']} - {s['title']}\n{s['text']}" for s in sections])
    prompt = f"""
You are a legal assistant. Based on the following sections of the {act_choice} Act, answer the question clearly and concisely.

Question: {query}

Relevant Sections:
{context}

Answer:
"""
    try:
        response = together.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error from LLaMA API: {e}"


def stream_text_animation(text_placeholder, text_content):
    displayed_text = ""
    for char in text_content:
        displayed_text += char
        answer_html = f"""
        <div class='answer-box'>
            <p>{displayed_text}</p>
        </div>
        """
        text_placeholder.markdown(answer_html, unsafe_allow_html=True)
        time.sleep(0.01)


if "chat" not in st.session_state:
    st.session_state.chat = []
for i, (role, msg) in enumerate(st.session_state.chat):
    with st.chat_message(role):
        st.markdown(msg)
query = st.chat_input("Ask You Question")
with footer_container:
    st.markdown("""
        <div class="footer">
            Created by <a href="https://www.linkedin.com/in/deepshah2712/" target="_blank">Deep Shah</a> and
            <a href="https://www.linkedin.com/in/rihaan-r-shaikh/" target="_blank">Rihaan Shaikh</a>
        </div>
    """, unsafe_allow_html=True)
if query:
    with footer_container:
        st.markdown("""
            <div class="footer">
                Created by <a href="https://www.linkedin.com/in/deepshah2712/" target="_blank">Deep Shah</a>
            </div>
        """, unsafe_allow_html=True)

    st.chat_message("user").markdown(query)
    st.session_state.chat.append(("user", query))

    with st.spinner("Searching legal knowledge base..."):
        top_sections = search_faiss(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ask_llama(query, top_sections)
        stream_text_animation(message_placeholder,full_response)

    st.session_state.chat.append(("ai", full_response))



