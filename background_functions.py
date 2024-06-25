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
from jobs_scrape import scrape_my_jobs

# Set Streamlit page configuration as the first command
st.set_page_config(page_title="CV Matcher", page_icon=":briefcase:")

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    st.warning(f"Error downloading NLTK data: {e}")

# Initialize Sentence-BERT model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Initialize KeyBERT model
kw_model = KeyBERT(model='paraphrase-distilroberta-base-v1')

# Initialize stop words
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    st.warning(f"Error loading stop words: {e}")
    stop_words = set()

# Load pre-trained SpaCy model for NER
nlp = spacy.load('en_core_web_sm')


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

def classify_cv_sections(text):
    # Initialize sections
    sections = {
        'education': [],
        'experience': [],
        'first_third': '',
        'second_third': ''
    }
    
    # Predefined keywords for education and experience
    keywords_education = [
        "education", "university", "college", "school", "bachelor’s", "master’s", "phd", 
        "associate degree", "diploma", "academic", "scholarship", "graduate", "undergraduate", 
        "mba", "b.sc", "m.sc", "ph.d", "coursework", "curriculum", "credits", "major", "minor", 
        "faculty", "department", "dean's list", "honors", "cum laude", "magna cum laude", 
        "summa cum laude", "thesis", "dissertation", "capstone", "internship", "training", 
        "certifications", "certified", "license", "accredited", "BSc", "MSc", "PhD", "BA", "MA",
        "BS", "MS", "engineering", "management", "science", "technology", "informatics","relevant courses",
        "relevant coursework", "gpa", "grade point average", "academic merit", "final capstone"

    ]

    keywords_education = keywords_education + [word.capitalize() for word in keywords_education]
    
    keywords_experience = [
        "experience", "career", "professional", "internship", "position", "role", "job", 
        "employment", "work", "project", "task", "duty", "responsibility", "achievement", 
        "accomplishment", "contribution", "involvement", "engagement", "assignment", 
        "managed", "developed", "created", "implemented", "led", "coordinated", "collaborated", 
        "participated", "supported", "maintained", "analyzed", "planned", "executed", 
        "evaluated", "improved", "optimized", "delivered", "resolved", "engineered", 
        "designed", "consulted", "programmed", "configured", "deployed", "tested", 
        "upgraded", "automated", "volunteer", "consultancy", "freelance", "full-time", 
        "part-time", "seasonal", "contract", "temporary", "tenure", "apprenticeship",
        "project", "team", "teamwork", "leadership", "management", "supervision",
        "communication", "problem-solving", "decision-making", "analytical", "technical",
        "stakeholder", "client", "customer", "business", "industry", "market", "strategy",
        "planning", "development", "implementation", "execution", "monitoring", "evaluation",
        "reporting", "presentation", "training", "mentoring", "coaching", "guidance",
        "support", "feedback", "review", "assessment", "audit", "quality", "performance",
        "process", "procedure", "methodology", "technology", "solution", "system",
        "software", "hardware", "database", "network", "cloud", "security", "architecture",
        "framework", "model", "algorithm", "tool", "language", "library", "platform",
        "environment", "standard", "requirement", "regulation", "compliance", "policy"
    ]

    keywords_experience = keywords_experience + [word.capitalize() for word in keywords_experience]
    
    # Process the document with spaCy
    doc = nlp(text)
    
    # Split the CV into thirds
    one_third_length = int(len(doc) / 3)
    sections['first_third'] = doc[:one_third_length].text
    sections['second_third'] = doc[one_third_length:].text

    # Iterate over sentences to classify them based on keywords
    for sent in doc.sents:
        sentence = sent.text.strip().lower()
        if any(kw.lower() in sentence for kw in keywords_education):
            sections['education'].append(sent.text.strip())
        elif any(kw.lower() in sentence for kw in keywords_experience):
            sections['experience'].append(sent.text.strip())

    # Join sentences for education and experience sections
    for key in ['education', 'experience']:
        sections[key] = "\n".join(sections[key])

    return sections

def classify_job_description(text):
    # Initialize sections
    sections = {'requirements': [], 'responsibilities': [], 'rest': []}
    current_section = 'rest'
    
    # Union of all keywords for requirements and responsibilities
    keywords_requirements_union = [
        "education and experience required", "knowledge and skills", "additional skills", 
        "bachelor’s degree", "master’s degree", "knowledge of programming languages", 
        "conceptual understanding of", "passion for new and emerging", "willingness to participate",
        "communicate effectively", "problem-solving skills", "able to produce", "teamwork skills", 
        "autonomy", "active learning", "active listening", "client expectations management",
        "at least 2 years of experience", "experience as a business analysis", "customer experience design roles", 
        "established quality standards", "industry benchmarks", "creating detailed and thorough as-is analysis", 
        "designing to-be processes", "understand the business requirements", "translate them into technical requirements", 
        "knowledge and experience of implementing", "proven experience of ba & change management", 
        "knowledge of agile methodology", "experience in alm tools", "experience with contact centre solutions", 
        "strong written and verbal communication skills", "certifications", "understanding of financial institutions",
        "experience as a business analyst", "experience with process modeling tools", "knowledge of jira and/or confluence tools",
        "fluency in microsoft office suite", "fluent in english", "Minimum 2 years ", "Minimum 3 years ", "Minimum 4 years ", 
        "Minimum 5 years ", "Minimum 6 years ", "Minimum 7 years ", "Minimum 8 years ", "Minimum 9 years ", "Minimum 10 years ",
        "years of experience", "experience in a similar role", "experience in a similar position", "experience in a similar environment",
        "looking for", "How about you?", "What we're looking for", "What we are looking for", "What we are looking for in you","what we are looking for in you",
        "what you'll bring", "what you will bring", "what you'll need", "what you will need", "what you'll do", "what you will do",
        "about you","Requirements", "requirements", "qualifications", "Qualifications", "skills", "Skills", "experience", "Experience","ability",
        "Ability","Knowledge", "knowledge", "Knowledgeable", "knowledgeable","diploma", "Academic degree", "academic degree", "certification", "Certification", 
        "certified", "Certified","Expertise", "expertise", "Expert", "expert", "Proficient", "proficient", "Competent", "competent", "Fluent", "fluent","Post-Bologna", 
        "post-Bologna", "Post Bologna", "post Bologna", "Post-Bologna degree", "post-Bologna degree", "Post Bologna degree", "post Bologna degree", "Master", "master",
        "Up to", "up to", "At least", "at least", "Minimum", "minimum", "Preferred", "preferred", "Plus", "plus", "Nice to have", "nice to have", "Desirable", "desirable",
        "have a Bachelor", "have a Master", "have a PhD", "have a Post-Bologna", "have a Post Bologna", "have a Post-Bologna degree", "have a Post Bologna degree"," have a Master's in a field",
        "have a Bachelor's in a field", "have a PhD in a field", "have a Post-Bologna in a field", "have a Post Bologna in a field", "have a Post-Bologna degree in a field",
        "have a bachelor's and a master's degree", "have a bachelor's and a PhD", "have a bachelor's and a Post-Bologna", "have a bachelor's and a Post Bologna",
        "what will make you succeed", "what will make you successful", "what will make you triumph", "what will make you win", "what will make you stand out",
        "expertise in", "Expertise in", "experience with", "Experience with", "knowledge of", "Knowledge of", "skills in", "Skills in", "ability to", "Ability to",
        "Good knowledge of", "good knowledge of", "Strong knowledge of", "strong knowledge of", "Excellent knowledge of", "excellent knowledge of", "Solid knowledge of", "solid knowledge of"
    ]
    keywords_responsibilities_union = [
        "responsible for delivery", "work under supervision", "integrate technical knowledge", 
        "apply company solutions", "provide technical consulting", "resolve technical issues", 
        "manage smaller projects/programs", "act professionally as trusted advisor", 
        "participates as part of a team", "understand the company's strategy", "keep refreshing technical knowledge", 
        "participate in technical or professional community events",
        "leading transformation and bau activities", "join our dynamic team", "recognising business requirements", 
        "analysing user stories", "managing the backlog", "engaging with project team members and stakeholders", 
        "ensure business requirements are captured", "ensuring timely and consistent communication", 
        "remain up to date with new practices",
        "responsible for working with company data", "reporting metrics", "analyzing methodologies", 
        "suggesting operation improvements", "building proposal evaluations", "in a cross-functional environment",
        "Develop", "develop", "Design", "design", "Create", "create","responsible for", "Responsible for", "responsible of", "Responsible of",
        "Process", "process", "Manage", "manage", "Lead", "lead", "Coordinate", "coordinate", "Collaborate", "collaborate", "Support", "support",
        "Implement", "implement", "Monitor", "monitor", "Analyze", "analyze", "Evaluate", "evaluate", "Define", "define", "Optimize", "optimize",
        "Ensure", "ensure", "Establish", "establish", "Maintain", "maintain", "Perform", "perform", "Participate", "participate", "Contribute", "contribute",
        "collect", "Collect", "prepare", "Prepare", "deliver", "Deliver", "report", "Report", "communicate", "Communicate", "engage", "Engage", "work", "Work",
        "collaborate", "Collaborate", "coordinate", "Coordinate", "support", "Support", "monitor", "Monitor", "analyze", "Analyze", "evaluate", "Evaluate",
        "report", "Report", "ensure", "Ensure", "maintain", "Maintain", "participate", "Participate", "contribute", "Contribute", "develop", "Develop",
        "identify","Identify" 
    ]
    
    # Process the document with spaCy
    doc = nlp(text)
    
    # Iterate over sentences to classify them based on keywords
    for sent in doc.sents:
        sentence = sent.text.strip().lower()
        if any(kw in sentence for kw in keywords_responsibilities_union):
            current_section = 'responsibilities'
        elif any(kw in sentence for kw in keywords_requirements_union):
            current_section = 'requirements'
        else:
            current_section = 'rest'
        
        sections[current_section].append(sent.text.strip())

    # Join sentences for each section
    for key in sections:
        sections[key] = "\n".join(sections[key])

    return sections


def find_similarity(text1, text2):
    """Calculate cosine similarity between two pieces of text."""
    embedding1 = embed_text(text1)
    embedding2 = embed_text(text2)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

# Define function to find top matches between CV and job descriptions
def find_top_matches(cv_text, job_descriptions, top_n=3):
    cv_sections = classify_cv_sections(cv_text)
    cv_education = cv_sections['education']
    cv_experience = cv_sections['experience']
    print(cv_education)
    print(cv_experience)

    job_sections = [classify_job_description(job) for job in job_descriptions]
    job_requirements = [" ".join(job['requirements']) for job in job_sections]
    job_responsibilities = [" ".join(job['responsibilities']) for job in job_sections]

    # Combine all job requirements and responsibilities for comparison
    combined_job_texts = [reqs + ' ' + resp for reqs, resp in zip(job_requirements, job_responsibilities)]

    # Calculate similarities for the given pairs
    education_job_similarities = [find_similarity(cv_education, text) for text in combined_job_texts]
    experience_job_similarities = [find_similarity(cv_experience, text) for text in combined_job_texts]
    education_requirements_similarities = [find_similarity(cv_education, req) for req in job_requirements]
    experience_responsibilities_similarities = [find_similarity(cv_experience, resp) for resp in job_responsibilities]
    overall_similarities = [find_similarity(cv_text, job) for job in job_descriptions]

    # Find top N matches for overall similarity
    def get_top_matches(similarities):
        return sorted([(score, index) for index, score in enumerate(similarities)], reverse=True)[:top_n]
    
    #get a list of the rest of the comparison types base on the index of the top matches
    def get_index_functions(list1,list2):
        print(list2)
        indexs = [element[1] for element in list2]
        print(indexs)
        top_matches = []
        for i in range(len(list1)):
            print(i)
            if i in indexs:
                top_matches += [(list1[i],i)]
        return top_matches
    
    ov_results = get_top_matches(overall_similarities)
    print(ov_results)

    results = {
        'education_job_similarities': get_index_functions(education_job_similarities,ov_results),
        'experience_job_similarities': get_index_functions (experience_job_similarities,ov_results),
        'education_requirements_similarities': get_index_functions(education_requirements_similarities,ov_results),
        'experience_responsibilities_similarities': get_index_functions(experience_responsibilities_similarities,ov_results),
        'overall_similarities': ov_results
    }

    return results

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


# Define function to enhance CV based on job description
def enhance_cv(cv_text, job_text):
    job_keywords = kw_model.extract_keywords(job_text, keyphrase_ngram_range=(1, 2), stop_words='english')
    cv_keywords = kw_model.extract_keywords(cv_text, keyphrase_ngram_range=(1, 2), stop_words='english')
    
    job_keywords_set = set([kw[0] for kw in job_keywords])
    cv_keywords_set = set([kw[0] for kw in cv_keywords])
    
    missing_keywords = job_keywords_set - cv_keywords_set
    suggestions = ', '.join(missing_keywords)
    
    return suggestions


def get_missing_keywords(cv_text, job_text):
    job_keywords = kw_model.extract_keywords(job_text, keyphrase_ngram_range=(1, 2), stop_words='english')
    cv_keywords = kw_model.extract_keywords(cv_text, keyphrase_ngram_range=(1, 2), stop_words='english')
    
    job_keywords_set = set([kw[0] for kw in job_keywords])
    cv_keywords_set = set([kw[0] for kw in cv_keywords])
    
    missing_keywords = job_keywords_set - cv_keywords_set
    return missing_keywords

def generate_interview_questions(job_text,openai_key):
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_key)
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