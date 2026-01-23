import streamlit as st
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from fpdf import FPDF

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Study Buddy (Offline & Free)",
    layout="wide"
)

# ---------------- LOAD LOCAL MODEL ----------------
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "google/flan-t5-base"  # FREE, LOCAL, STABLE
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- SESSION STATE ----------------
if "content_text" not in st.session_state:
    st.session_state.content_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📘 AI Study Buddy")
menu = st.sidebar.radio(
    "Navigation",
    ["Upload PDF", "Upload Notes", "Ask Questions"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🕘 Question History")
for q in st.session_state.history[::-1]:
    st.sidebar.write("•", q)

# ---------------- MAIN TITLE ----------------
st.title("📚 AI Study Buddy (FREE & OFFLINE)")
st.write(
    "This app explains study material **paragraph-wise** using a "
    "**local AI model** (no API, no cost)."
)

# ---------------- AI FUNCTION ----------------
def ask_ai(prompt, max_len=256):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=4,
            repetition_penalty=1.5,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- PDF READER ----------------
def read_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---------------- PARAGRAPH SPLITTER ----------------
def split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]

# ---------------- PARAGRAPH EXPLANATION ----------------
def explain_paragraphs(text):
    paragraphs = split_paragraphs(text)
    output = ""

    for i, para in enumerate(paragraphs, start=1):
        prompt = f"""
You are an expert academic tutor.
Explain the following paragraph clearly and simply.
Do NOT invent examples.
Do NOT add unrelated objects.
Explain only the meaning.

Paragraph:
{para}
"""
        explanation = ask_ai(prompt)

        output += (
            f"Paragraph {i}:\n"
            f"{para}\n\n"
            f"Explanation:\n{explanation}\n\n"
            f"{'-'*60}\n\n"
        )

    return output

# ================= UPLOAD PDF =================
if menu == "Upload PDF":
    st.subheader("📄 Upload PDF")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_pdf:
        text = read_pdf(uploaded_pdf)
        st.session_state.content_text = text

        st.success("✅ PDF uploaded successfully")

        with st.spinner("🧠 Explaining paragraph-wise..."):
            explanation = explain_paragraphs(text)

        st.subheader("📘 Paragraph-wise Explanation")
        st.text_area("Output", explanation, height=550)

        if st.button("📥 Download Explanation PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=11)

            for line in explanation.split("\n"):
                pdf.multi_cell(0, 7, line)

            pdf.output("AI_Study_Buddy_Explanation.pdf")
            st.success("📄 PDF downloaded successfully")

# ================= UPLOAD NOTES =================
elif menu == "Upload Notes":
    st.subheader("📝 Paste Notes")
    notes = st.text_area("Paste your notes here", height=250)

    if st.button("Explain Notes"):
        if not notes.strip():
            st.warning("Please paste notes first.")
        else:
            st.session_state.content_text = notes

            with st.spinner("🧠 Explaining paragraph-wise..."):
                explanation = explain_paragraphs(notes)

            st.subheader("📘 Explanation")
            st.text_area("Output", explanation, height=550)

# ================= ASK QUESTIONS =================
elif menu == "Ask Questions":
    st.subheader("❓ Ask Questions")

    if not st.session_state.content_text:
        st.warning("Upload PDF or Notes first.")
    else:
        question = st.text_area("Enter your question")

        if st.button("Get Answer"):
            st.session_state.history.append(question)

            prompt = f"""
Answer the question using ONLY the content below.
If the answer is not present, say:
"Answer not found in the provided material."

Content:
{st.session_state.content_text}

Question:
{question}
"""
            with st.spinner("🤖 Thinking..."):
                answer = ask_ai(prompt)

            st.subheader("✅ Answer")
            st.write(answer)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>✨ Developed by <b>Saika Parvin</b></center>",
    unsafe_allow_html=True
)
