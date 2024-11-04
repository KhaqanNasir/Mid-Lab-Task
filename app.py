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
        cv_scores = analyze_cv_text(cv)  # Get scores for each CV
        skills_score = cv_scores[0][0] * 100  # Convert to percentage
        experience_score = cv_scores[0][1] * 100  # Convert to percentage

        total_score = (0.3 * skills_score + 0.3 * experience_score)  # Adjust weights as necessary
        all_candidates.append((cv, total_score, skills_score, experience_score))

    return all_candidates

# Streamlit UI
st.set_page_config(page_title="CV Analysis Tool", layout="wide")  # Set page title and layout
st.markdown("""
    <h2 style='text-align: center; color: white;'>Developed by Muhammad Khaqan Nasir</h2>
    <p style='text-align: center; '>
        <a href='https://www.linkedin.com/in/khaqan-nasir/' target='_blank'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' alt='LinkedIn' width='24' style='vertical-align: middle; margin-right: 8px;'/>
            LinkedIn
        </a>
    </p>
    """, unsafe_allow_html=True)
st.title("🌟 CV Analysis and Candidate Ranking Tool 🌟")
st.markdown("## Upload your CVs for analysis and receive scores based on skills and experience!")

# Upload the CV PDF files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, help="Upload your CVs in PDF format")

if uploaded_files:
    # Read and analyze the CVs
    st.spinner("Extracting text from the CVs...")
    cvs = [extract_text_from_pdf(uploaded_file) for uploaded_file in uploaded_files]  # Extract text from all uploaded PDFs

    # Rank candidates
    candidates = rank_candidates(cv_data=cvs)

    # Display candidates
    if candidates:
        st.subheader("Candidates Analysis")
        for index, candidate in enumerate(candidates):
            cv, total_score, skills_score, experience_score = candidate
            st.write(f"**CV Snippet {index + 1}:** {cv[:30]}...")  # Print a snippet of the CV text
            st.write(f"**Total Score:** {total_score:.2f}%")
            st.write(f"**Skills Score:** {skills_score:.2f}%")
            st.write(f"**Experience Score:** {experience_score:.2f}%")

        # Plotting the scores for each CV
        st.subheader("Comparison of Skills and Experience Scores")
        
        labels = [f"CV {i + 1}" for i in range(len(candidates))]
        skills_scores = [candidate[2] for candidate in candidates]
        experience_scores = [candidate[3] for candidate in candidates]

        x = range(len(candidates))

        fig, ax = plt.subplots()
        ax.bar(x, skills_scores, width=0.4, label='Skills Score', color='#1f77b4', align='center')
        ax.bar([p + 0.4 for p in x], experience_scores, width=0.4, label='Experience Score', color='#ff7f0e', align='center')

        ax.set_ylim(0, 100)
        ax.set_xticks([p + 0.2 for p in x])
        ax.set_xticklabels(labels)
        ax.set_ylabel("Percentile (%)")
        ax.set_title("Comparison of Candidate Skills and Experience Scores")
        ax.legend()

        st.pyplot(fig)

        # Determine the best candidate
        best_candidate = max(candidates, key=lambda x: x[1])
        best_cv, best_total_score, best_skills_score, best_experience_score = best_candidate
        st.markdown(f"### Best Candidate: CV Snippet: {best_cv[:30]}...")
        st.write(f"**Best Total Score:** {best_total_score:.2f}%")
        st.write(f"**Best Skills Score:** {best_skills_score:.2f}%")
        st.write(f"**Best Experience Score:** {best_experience_score:.2f}%")
    else:
        st.warning("No candidates found. Please check the CV content.")

# Clear button to reset uploaded CVs
if st.button("Clear CVs"):
    if 'uploaded_files' in st.session_state:
        del st.session_state.uploaded_files  # Clear the uploaded files
    st.experimental_rerun()

# Footer
st.markdown("""
---
*This CV analysis tool helps you evaluate your qualifications based on your CV. Ensure your CV is well-formatted and clear for the best results!*
""")
