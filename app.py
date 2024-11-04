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
st.title("CV Analysis and Candidate Ranking")

# Upload the CV PDF file
uploaded_file = st.file_uploader("Upload CV PDF", type="pdf")

if uploaded_file:
    # Read and analyze the CV
    cv_text = extract_text_from_pdf(uploaded_file)  # Extract text from the PDF CV
    cvs = [cv_text]  # List of CV texts for analysis

    # Rank candidates
    candidates = rank_candidates(cv_data=cvs)

    # Display candidates
    if candidates:
        st.subheader("Candidates Analysis")
        for candidate in candidates:
            cv, total_score, skills_score, experience_score = candidate
            st.write(f"\nCV Snippet: {cv[:30]}...")  # Print a snippet of the CV text
            st.write(f"Total Score: {total_score:.2f}%")
            st.write(f"Skills Score: {skills_score:.2f}%")
            st.write(f"Experience Score: {experience_score:.2f}%")

        # Plotting the skills and experience scores
        st.subheader("Skills and Experience Scores")
        labels = ['Skills Score', 'Experience Score']
        scores = [candidates[0][2], candidates[0][3]]  # Get scores from the first candidate

        fig, ax = plt.subplots()
        ax.bar(labels, scores, color=['blue', 'orange'])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentile (%)")
        ax.set_title("Candidate Skills and Experience Scores")
        
        st.pyplot(fig)  # Display the plot in Streamlit

    else:
        st.write("No candidates found.")
