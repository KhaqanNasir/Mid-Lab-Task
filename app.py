import pdfplumber
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import streamlit as st

# Load the pretrained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Function to extract text from a PDF CV
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
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

        # Check if the candidate is suitable based on thresholds
        is_suitable = skills_score >= 30 and experience_score >= 30  # Threshold set to 30%
        total_score = (0.3 * skills_score + 0.3 * experience_score)  # Adjust weights as necessary

        all_candidates.append((cv, total_score, skills_score, experience_score, is_suitable))

    return all_candidates

# Streamlit app layout
st.title("CV Analysis Tool")
st.write("Upload your CV in PDF format to analyze its content and receive scores based on skills and experience.")

# Upload CV file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF CV
    cv_text = extract_text_from_pdf(uploaded_file)
    cvs = [cv_text]  # List of CV texts for analysis

    # Rank candidates
    candidates = rank_candidates(cv_data=cvs)

    # Display candidates
    if candidates:
        st.subheader("Analysis Results:")
        for candidate in candidates:
            cv, total_score, skills_score, experience_score, is_suitable = candidate
            st.write(f"**CV Snippet:** {cv[:30]}...")  # Print a snippet of the CV text
            st.write(f"**Total Score:** {total_score:.2f}%")
            st.write(f"**Skills Score:** {skills_score:.2f}%")
            st.write(f"**Experience Score:** {experience_score:.2f}%")
            st.write("**Status:** Suitable Candidate" if is_suitable else "Not a Suitable Candidate")
    else:
        st.write("No candidates found.")
