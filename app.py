import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from fpdf import FPDF

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Study Buddy (Line-by-Line PDF Explanation)",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "google/flan-t5-small"  # lightweight; change to flan-t5-base for longer notes
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
st.sidebar.subheader("🕘 Search History")
if st.session_state.history:
    for q in st.session_state.history[::-1]:
        st.sidebar.write("•", q)
else:
    st.sidebar.write("No searches yet")

# ---------------- MAIN TITLE ----------------
st.title("📚 AI Study Buddy")
st.write("Upload study material. AI explains **every line** in detail for full understanding.")

# ---------------- AI FUNCTION ----------------
def ask_ai(prompt, max_len=512):
    """Get detailed answer from local Flan-T5 model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_len)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ---------------- PDF READER ----------------
def read_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---------------- LINE-BY-LINE EXPLANATION ----------------
def explain_line_by_line(text):
    """Explain every line of text in detail"""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    detailed_notes = ""
    for idx, line in enumerate(lines):
        # Prompt AI to explain a single line in long notes form
        prompt = f"Explain this line in detail for a student. Use simple words and give examples if needed:\n\n{line}"
        explanation = ask_ai(prompt)
        detailed_notes += f"Line {idx+1}: {line}\nExplanation: {explanation}\n\n"
    return detailed_notes

# ================= UPLOAD PDF =================
if menu == "Upload PDF":
    st.subheader("📄 Upload PDF")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_pdf:
        text = read_pdf(uploaded_pdf)
        st.session_state.content_text = text
        st.success(f"✅ PDF uploaded successfully ({len(text.split())} words)")

        with st.spinner("📖 AI is explaining PDF line by line..."):
            full_explanation = explain_line_by_line(text)

        st.subheader("🧠 Detailed PDF Explanation")
        st.write(full_explanation)

        # ---------------- DOWNLOAD PDF ----------------
        if st.button("Download PDF Notes"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in full_explanation.split("\n"):
                pdf.multi_cell(0, 8, line)
            pdf.output("pdf_detailed_notes.pdf")
            st.success("📄 Notes saved as pdf_detailed_notes.pdf")

# ================= UPLOAD NOTES =================
elif menu == "Upload Notes":
    st.subheader("📝 Paste Your Notes")
    notes = st.text_area(
        "Paste your notes here",
        height=250,
        placeholder="Paste handwritten or typed notes..."
    )

    if st.button("Explain Notes Line by Line"):
        if notes.strip() == "":
            st.warning("Please paste notes first.")
        else:
            st.session_state.content_text = notes
            with st.spinner("📘 AI is explaining your notes line by line..."):
                detailed_notes = explain_line_by_line(notes)
            st.subheader("🧠 Detailed Notes Explanation")
            st.write(detailed_notes)

            # Download PDF
            if st.button("Download Notes PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in detailed_notes.split("\n"):
                    pdf.multi_cell(0, 8, line)
                pdf.output("notes_detailed.pdf")
                st.success("📄 Notes saved as notes_detailed.pdf")

# ================= ASK QUESTIONS =================
elif menu == "Ask Questions":
    st.subheader("❓ Ask Questions from Material")
    if st.session_state.content_text == "":
        st.warning("Upload PDF or Notes first.")
    else:
        question = st.text_area("Ask your question")
        if st.button("Get Answer"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                st.session_state.history.append(question)
                with st.spinner("🤖 AI is answering..."):
                    prompt = f"""
Answer the question using ONLY the content below. 
If answer is not found, say so clearly.

Content:
{st.session_state.content_text}

Question:
{question}
"""
                    answer = ask_ai(prompt)
                st.subheader("✅ Answer")
                st.write(answer)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>✨ Developed by <b>Saika Parvin</b></center>",
    unsafe_allow_html=True
)
