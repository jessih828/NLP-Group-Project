import os
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.corpus import stopwords
import re
import spacy
import streamlit as st
from io import BytesIO
import time
from keybert import KeyBERT
from openai import OpenAI

# Set Streamlit page configuration as the first command
st.set_page_config(page_title="CV Matcher", page_icon=":briefcase:")

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    st.warning(f"Error downloading NLTK data: {e}")

# Initialize Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize KeyBERT model
kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')

# Initialize stop words
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    st.warning(f"Error loading stop words: {e}")
    stop_words = set()

# Load pre-trained SpaCy model for NER

import subprocess

# Function to download and load SpaCy model
def download_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
    except OSError:
        st.info(f"Downloading SpaCy model {model_name}...")
        try:
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            nlp = spacy.load(model_name)
        except subprocess.CalledProcessError as e:
            st.error(f"Error downloading SpaCy model: {e}")
            raise
    return nlp

# Load pre-trained SpaCy model for NER
try:
    nlp = download_spacy_model('en_core_web_sm')
except Exception as e:
    st.error(f"Failed to load SpaCy model: {e}")

# Define a simple list of skills for demonstration purposes
skill_set = {"python", "data analysis", "machine learning", "project management", "communication"}

# Define function to extract entities using NER
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE', 'NORP', 'FAC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE']]
    return ' '.join(entities)

# Define function to extract skills from text
def extract_skills(text):
    words = set(text.split())
    skills = skill_set.intersection(words)
    return ' '.join(skills)

# Define function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    
    # Extract entities and skills
    entities = extract_entities(cleaned_text)
    skills = extract_skills(cleaned_text)
    
    return cleaned_text + ' ' + entities + ' ' + skills

# Define function to extract text from a PDF file
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            text += page.extract_text() + " "
    return text.strip()

# Define function to embed text using Sentence-BERT
def embed_text(text):
    return model.encode(text, convert_to_tensor=True)

# Define function to find top matches between CV and job descriptions
def find_top_matches(cv_text, job_descriptions, top_n=3):
    cv_embedding = embed_text(cv_text)
    job_embeddings = [embed_text(desc) for desc in job_descriptions]
    
    similarity_scores = [util.pytorch_cos_sim(cv_embedding, job_embedding).item() for job_embedding in job_embeddings]
    top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:top_n]
    top_matches = [(similarity_scores[index], index) for index in top_indices]
    
    return top_matches

# Define function to calculate similarity between CV and a single job description
def calculate_similarity(cv_text, job_text):
    cv_embedding = embed_text(cv_text)
    job_embedding = embed_text(job_text)
    similarity_score = util.pytorch_cos_sim(cv_embedding, job_embedding).item()
    return similarity_score

# Define function to load and prepare job descriptions data
def load_and_prepare_data(csv_path):
    jobs_df = pd.read_csv(csv_path)
    jobs_df.dropna(subset=['company_name', 'title', 'description'], inplace=True)
    return jobs_df

# Define function to sample job descriptions
def sample_job_descriptions(jobs_df, sample_size):
    sampled_jobs_df = jobs_df.sample(min(sample_size, len(jobs_df)), random_state=1)
    sampled_jobs_df['description'] = sampled_jobs_df['description'].apply(preprocess_text)
    return sampled_jobs_df

# Define function to enhance CV based on job description
def enhance_cv(cv_text, job_text):
    job_keywords = kw_model.extract_keywords(job_text, keyphrase_ngram_range=(1, 2), stop_words='english')
    cv_keywords = kw_model.extract_keywords(cv_text, keyphrase_ngram_range=(1, 2), stop_words='english')
    
    job_keywords_set = set([kw[0] for kw in job_keywords])
    cv_keywords_set = set([kw[0] for kw in cv_keywords])
    
    missing_keywords = job_keywords_set - cv_keywords_set
    suggestions = ', '.join(missing_keywords)
    
    return suggestions

def get_missing_skills(cv_text, job_text):
    job_skills = extract_skills(job_text)
    cv_skills = extract_skills(cv_text)
    
    missing_skills = set(job_skills.split()) - set(cv_skills.split())
    return missing_skills

def get_missing_keywords(cv_text, job_text):
    job_keywords = kw_model.extract_keywords(job_text, keyphrase_ngram_range=(1, 2), stop_words='english')
    cv_keywords = kw_model.extract_keywords(cv_text, keyphrase_ngram_range=(1, 2), stop_words='english')
    
    job_keywords_set = set([kw[0] for kw in job_keywords])
    cv_keywords_set = set([kw[0] for kw in cv_keywords])
    
    missing_keywords = job_keywords_set - cv_keywords_set
    return missing_keywords

from openai import OpenAI

api_key = os.getenv('API_KEY')
# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def generate_interview_questions(job_text):
    prompt = f"Generate 5 interview questions for a job posting with the following description:\n\n{job_text}\n\nInterview Questions:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    # Access the message content directly from the response
    questions = response.choices[0].message.content.strip().split('\n')
    return questions

# Page for comparing a CV against a dataset of job postings
def page_dataset_comparison():
    st.image('image_banner.png', use_column_width=True)
    st.title("Looking for a job? Let's find the best match!")
    st.write("Upload your CV and find the top 3 matching job descriptions from our dataset.")
    
    sample_size = st.number_input("Enter the number of random jobs to consider (bigger sample size takes longer):", min_value=1, value=100, step=1, key="sample_size")
    st.write("Note: The larger the sample size, the longer it will take to process.")
    
    uploaded_cv = st.file_uploader("Choose a PDF file", type="pdf", key="cv")
    
    if uploaded_cv is not None:
        with st.spinner('Processing...'):
            start_time = time.time()
            
            # Read the uploaded CV file
            cv_text = preprocess_text(extract_text_from_pdf(uploaded_cv))

            # Load and prepare job descriptions data
            csv_path = 'postings.csv'
            jobs_df = load_and_prepare_data(csv_path)

            # Sample job descriptions
            sampled_jobs_df = sample_job_descriptions(jobs_df, sample_size)

            # Find top matches
            top_matches = find_top_matches(cv_text, sampled_jobs_df['description'].tolist(), top_n=3)

            end_time = time.time()
            elapsed_time = end_time - start_time
        
        # Display top matches
        st.write(f"Processing completed in {elapsed_time:.2f} seconds")
        st.write("Top 3 matching job descriptions:")
        for i, (score, index) in enumerate(top_matches, start=1):
            company_name = sampled_jobs_df.iloc[index]['company_name']
            title = sampled_jobs_df.iloc[index]['title']
            description = sampled_jobs_df.iloc[index]['description']
            st.markdown(
                f"""
                <div class="result-card">
                    <h3>{i}. {title} at {company_name}</h3>
                    <p><strong>Similarity:</strong> {score * 100:.2f}%</p>
                    <p><strong>Description:</strong> {description[:200]}...</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

# Page for comparing a CV against a single job posting
def page_single_comparison():
    st.image('image_banner.png', use_column_width=True)
    st.title("CV vs Job Posting Comparison")
    st.write("Upload your CV and a job posting to see if you're a good fit.")
    
    uploaded_cv = st.file_uploader("Choose your CV (PDF)", type="pdf", key="cv_single")
    uploaded_job = st.file_uploader("Choose the job posting (PDF)", type="pdf", key="job_single")
    
    if st.button("Compare"):
        if uploaded_cv is not None and uploaded_job is not None:
            with st.spinner('Processing...'):
                # Read the uploaded CV and job posting files
                cv_text = preprocess_text(extract_text_from_pdf(uploaded_cv))
                job_text = preprocess_text(extract_text_from_pdf(uploaded_job))

                # Calculate similarity
                similarity_score = calculate_similarity(cv_text, job_text)
                
                # Display result
                if similarity_score >= 0.75:
                    st.success(f"You're a good fit! (Similarity: {similarity_score * 100:.2f}%)")
                else:
                    st.warning(f"I don't know, maybe try something else? (Similarity: {similarity_score * 100:.2f}%)")
        else:
            st.warning("Please upload both your CV and the job posting.")

# Page for enhancing CV based on job posting
def page_enhance_cv():
    st.image('image_banner.png', use_column_width=True)
    st.title("Enhance my CV")
    st.write("Upload your CV and the job posting to get suggestions on how to improve your CV to match the job posting.")
    
    uploaded_cv = st.file_uploader("Choose your CV (PDF)", type="pdf", key="cv_enhance")
    uploaded_job = st.file_uploader("Choose the job posting (PDF)", type="pdf", key="job_enhance")
    
    if st.button("Enhance"):
        if uploaded_cv is not None and uploaded_job is not None:
            with st.spinner('Processing...'):
                # Read the uploaded CV and job posting files
                cv_text = preprocess_text(extract_text_from_pdf(uploaded_cv))
                job_text = preprocess_text(extract_text_from_pdf(uploaded_job))

                # Enhance CV
                missing_skills = get_missing_skills(cv_text, job_text)
                missing_keywords = get_missing_keywords(cv_text, job_text)
                
                # Display result
                st.write(f"Suggestions to enhance your CV:")
                if missing_skills:
                    st.success(f"Missing Skills: {', '.join(missing_skills)}")
                else:
                    st.info("Your CV already contains all the required skills.")
                
                if missing_keywords:
                    st.success(f"Keywords to Add: {', '.join(missing_keywords)}")
                else:
                    st.info("Your CV already contains all the important keywords.")

# Page for generating interview questions based on job description
def page_interview_questions():
    st.image('image_banner.png', use_column_width=True)
    st.title("Interview Question Generator")
    st.write("Upload a job posting to generate potential interview questions based on the job description.")
    
    uploaded_job = st.file_uploader("Choose the job posting (PDF)", type="pdf", key="job_questions")
    
    if st.button("Generate Questions"):
        if uploaded_job is not None:
            with st.spinner('Processing...'):
                # Read the uploaded job posting file
                job_text = preprocess_text(extract_text_from_pdf(uploaded_job))

                # Generate interview questions
                interview_questions = generate_interview_questions(job_text)

                # Display result
                st.write("Generated Interview Questions:")
                for question in interview_questions:
                    st.write(f"- {question}")
        else:
            st.warning("Please upload the job posting.")

# Streamlit app
def main():
    # Add custom CSS for styling
    st.markdown('''
        <style>
            .css-1egvi7u {margin-top: -4rem;}
            .css-znku1x a {color: #9d03fc;}
            .css-qrbaxs {min-height: 0.0rem;}
            .stSpinner > div > div {border-top-color: #9d03fc;}
            .css-15tx938{min-height: 0.0rem;}
            header {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .sidebar-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                height: 100%;
                justify-content: center;
            }
            .stButton>button {
                width: 100%;
                padding: 0.5rem 1rem;
                margin-bottom: 0.5rem;
                border-radius: 4px;
                text-align: center;
                background-color: #ffffff;
                color: #000000;
                box-sizing: border-box;
            }
            .stButton>button:hover {
                background-color: #9d03fc;
                color: white;
            }
            .stButton>button:active {
                background-color: #9d03fc;
                color: white;
            }
        </style>
    ''', unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'Dataset Comparison'
    
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    if st.sidebar.button('Dataset Comparison'):
        st.session_state.page = 'Dataset Comparison'
    if st.sidebar.button('1-vs-1 Comparison'):
        st.session_state.page = '1-vs-1 Comparison'
    if st.sidebar.button('Enhance my CV'):
        st.session_state.page = 'Enhance my CV'
    if st.sidebar.button('Interview Question Generator'):
        st.session_state.page = 'Interview Question Generator'
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.page == "Dataset Comparison":
        page_dataset_comparison()
    elif st.session_state.page == "1-vs-1 Comparison":
        page_single_comparison()
    elif st.session_state.page == "Enhance my CV":
        page_enhance_cv()
    elif st.session_state.page == "Interview Question Generator":
        page_interview_questions()

if __name__ == "__main__":
    main()
