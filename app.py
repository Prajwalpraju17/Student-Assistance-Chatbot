from huggingface_hub import InferenceClient
# Example: Direct use of InferenceClient for text generation
def hf_direct_text_generation(prompt):
    client = InferenceClient(model="google/flan-t5-base")
    response = client.text_generation(prompt=prompt)
    return response.generated_text if hasattr(response, 'generated_text') else response
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv("app.env")

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
import tiktoken

# -------------------- LOAD API KEY --------------------
# Remove OpenAI key check
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     st.error("OPENAI_API_KEY not found. Please set it in app.env or as an environment variable.")
#     st.stop()

# Add Hugging Face key check
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN not found. Please set it in .env or as an environment variable.")
    st.stop()

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Student Assistance RAG Chatbot", layout="wide")
st.title("📚 Student Assistance Agentic RAG Chatbot")
st.write("✅ 24/7 academic support • ✅ RAG answering • ✅ Personalized study plan")

# -------------------- SESSION STATE --------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "profile" not in st.session_state:
    st.session_state.profile = {
        "subject": "Python",
        "level": "Beginner",
        "goal": "Crack Interviews",
        "daily_hours": 2
    }

# -------------------- SIDEBAR PROFILE --------------------
st.sidebar.header("👤 Student Profile")

st.session_state.profile["subject"] = st.sidebar.selectbox(
    "Select Subject",
    ["Python", "DSA", "DBMS", "Machine Learning", "Deep Learning", "Computer Networks", "OS", "Generative AI"]
)

st.session_state.profile["level"] = st.sidebar.selectbox(
    "Level",
    ["Beginner", "Intermediate", "Advanced"]
)

st.session_state.profile["goal"] = st.sidebar.selectbox(
    "Goal",
    ["Pass Exams", "Crack Interviews", "Build Projects", "Improve Coding"]
)

st.session_state.profile["daily_hours"] = st.sidebar.slider(
    "Daily Study Hours",
    1, 8, 2
)

# -------------------- UPLOAD PDF --------------------
st.sidebar.header("📄 Upload Study PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# -------------------- FUNCTIONS --------------------
def build_vectorstore(files):
    all_docs = []

    for file in files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        all_docs.extend(loader.load())

    # Use TokenTextSplitter for better alignment with OpenAI tokenization
    splitter = TokenTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        encoding_name="cl100k_base"  # Encoder for OpenAI models
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)

    return vectordb


def rag_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)  # <-- updated line

    context = "\n\n".join([d.page_content for d in docs])

    profile = st.session_state.profile

    system_prompt = f"""
You are a Student Assistance AI chatbot for technical students.
Your job:
1) Answer using ONLY the given Context.
2) If answer not in context, give conceptual explanation + tell user it was not in PDF.
3) Give examples based on student level.
4) Recommend next topics/resources based on goal and subject.

Student Profile:
Subject: {profile["subject"]}
Level: {profile["level"]}
Goal: {profile["goal"]}
Daily Study Hours: {profile["daily_hours"]}
"""

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        task="text2text-generation",
        temperature=0.2
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
    ]

    reply = llm.invoke(messages)
    return reply.content


def generate_study_plan():
    profile = st.session_state.profile

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        task="text2text-generation",
        temperature=0.2,
        max_new_tokens=256
    )

    prompt = f"""
Create a personalized 7-day study plan for a technical student.

Subject: {profile["subject"]}
Level: {profile["level"]}
Goal: {profile["goal"]}
Daily study time: {profile["daily_hours"]} hours/day

Format must be:
Day 1:
- Topics:
- Practice:
- Mini Task:
Day 2:
...

Make it practical and easy to follow.
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    return res.content


# -------------------- CREATE VECTOR DB --------------------
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("✅ Building Knowledge Base from PDFs..."):
        st.session_state.vectorstore = build_vectorstore(uploaded_files)
    st.success("✅ Study materials indexed successfully! Now ask questions.")

# -------------------- CHAT INPUT --------------------
st.subheader("💬 Ask Your Question")

question = st.text_input("Type your technical doubt here...")

col1, col2 = st.columns(2)
ask_btn = col1.button("🔍 Ask")
plan_btn = col2.button("📅 Generate Study Plan")

# -------------------- QUESTION ANSWERING --------------------
if ask_btn and question:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload at least one PDF first.")
    else:
        ans = rag_answer(question, st.session_state.vectorstore)
        st.session_state.chat.append(("You", question))
        st.session_state.chat.append(("Bot", ans))

# -------------------- STUDY PLAN --------------------
if plan_btn:
    plan = generate_study_plan()
    st.session_state.chat.append(("Bot", "✅ Personalized 7-Day Study Plan:"))
    st.session_state.chat.append(("Bot", plan))

# -------------------- DISPLAY CHAT --------------------
st.markdown("## 🧾 Chat History")
for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")