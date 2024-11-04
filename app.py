import pdfplumber
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import streamlit as st
import matplotlib.pyplot as plt

# Load the pretrained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Function to extract text from a PDF CV
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to analyze the CV text and return scores
def analyze_cv_text(cv_text):
    inputs = tokenizer(cv_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).numpy()
    return probabilities

# Function to rank candidates based on their CV analysis
def rank_candidates(cv_data):
    all_candidates = []
    for cv in cv_data:
        cv_scores = analyze_cv_text(cv)
        skills_score = cv_scores[0][0] * 100
        experience_score = cv_scores[0][1] * 100
        total_score = (0.5 * skills_score + 0.5 * experience_score)
        all_candidates.append((cv, total_score, skills_score, experience_score))
    return all_candidates

# Streamlit UI
st.set_page_config(page_title="Professional CV Analysis Tool", layout="wide")
st.title("🔍 Professional CV Analysis and Candidate Ranking Tool")
st.markdown("<h2 style='color: #2c3e50;'>Upload CVs to analyze skills and experience</h2>", unsafe_allow_html=True)
st.markdown("---")

# Upload the CV PDF files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, help="Upload CVs in PDF format")

# Clear button to reset uploaded CVs
if st.button("Clear CVs"):
    uploaded_files = None  # Reset the uploaded files
    st.experimental_rerun()

if uploaded_files:
    st.spinner("Extracting text and analyzing CVs...")
    cvs = [extract_text_from_pdf(uploaded_file) for uploaded_file in uploaded_files]
    
    # Rank candidates
    candidates = rank_candidates(cv_data=cvs)

    # Display candidates
    if candidates:
        st.markdown("<h3 style='color: #16a085;'>Candidates Analysis</h3>", unsafe_allow_html=True)
        for index, candidate in enumerate(candidates):
            cv, total_score, skills_score, experience_score = candidate
            st.write(f"<b>CV Snippet {index + 1}</b>: {cv[:30]}...", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #2980b9;'>Total Score:</span> {total_score:.2f}%", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #27ae60;'>Skills Score:</span> {skills_score:.2f}%", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #e74c3c;'>Experience Score:</span> {experience_score:.2f}%", unsafe_allow_html=True)

        # Plotting the scores for each CV
        st.markdown("<h3 style='color: #34495e;'>Comparison of Skills and Experience Scores</h3>", unsafe_allow_html=True)

        labels = [f"CV {i + 1}" for i in range(len(candidates))]
        skills_scores = [candidate[2] for candidate in candidates]
        experience_scores = [candidate[3] for candidate in candidates]

        x = range(len(candidates))

        # Styling the graph with professional colors
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, skills_scores, width=0.4, label='Skills Score', color='#3498db', align='center')
        ax.bar([p + 0.4 for p in x], experience_scores, width=0.4, label='Experience Score', color='#e67e22', align='center')

        ax.set_ylim(0, 100)
        ax.set_xticks([p + 0.2 for p in x])
        ax.set_xticklabels(labels)
        ax.set_ylabel("Percentile (%)", fontsize=12)
        ax.set_title("Candidate Skills and Experience Score Comparison", fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

        # Determine the best candidate
        best_candidate = max(candidates, key=lambda x: x[1])
        best_cv, best_total_score, best_skills_score, best_experience_score = best_candidate
        st.markdown(f"<h3 style='color: #8e44ad;'>Best Candidate:</h3> <b>CV Snippet:</b> {best_cv[:30]}...", unsafe_allow_html=True)
        st.markdown(f"<span style='color: #2ecc71;'>Best Total Score:</span> {best_total_score:.2f}%", unsafe_allow_html=True)
        st.markdown(f"<span style='color: #3498db;'>Best Skills Score:</span> {best_skills_score:.2f}%", unsafe_allow_html=True)
        st.markdown(f"<span style='color: #e74c3c;'>Best Experience Score:</span> {best_experience_score:.2f}%", unsafe_allow_html=True)
    else:
        st.warning("No valid candidates found. Please check the CV content.")
else:
    st.info("Upload one or more PDF CVs to begin the analysis.")

# Footer
st.markdown("""
<hr>
<p style="text-align: center; color: gray;">
*This tool is designed for professional CV analysis and candidate ranking. Please ensure your CVs are clear and well-formatted for accurate results.*
</p>
""", unsafe_allow_html=True)
