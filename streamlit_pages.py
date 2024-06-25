import streamlit as st
from background_functions import  extract_text_from_pdf, calculate_similarity, find_top_matches, get_missing_keywords, generate_interview_questions, classify_cv_sections, classify_job_description
import time
from jobs_scrape import scrape_my_jobs
import pandas as pd

def display_job_match(index, job_details, similarities):
    # Define styles directly in the elements to ensure they are applied
    card_style = "background-color: #f9f9f9; border-radius: 10px; padding: 20px; margin-top: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
    header_style = "margin-top: 0;"
    paragraph_style = "margin: 5px 0;"


    # Render the HTML with inline styles
    st.markdown(f"""
        <div style="{card_style}">
            <h3 style="{header_style}">{index}. {job_details['title']} </h3>
            <p style="{paragraph_style}"><strong>Overall Similarity:</strong> {similarities['overall_similarity'] * 100:.2f}%</p>
            <p style="{paragraph_style}"><strong>Education & Requirements Similarity:</strong> {similarities['education_requirements_similarity'] * 100:.2f}%</p>
            <p style="{paragraph_style}"><strong>Experience & Responsibilities Similarity:</strong> {similarities['experience_responsibilities_similarity'] * 100:.2f}%</p>
            <p style="{paragraph_style}"><strong>Description:</strong> {job_details['description'][:1000]}...</p>
        </div>
        """, unsafe_allow_html=True)


def page_dataset_comparison():
    st.image('image_banner.png', use_column_width=True)
    st.title("Looking for a job? Let's find the best match!")
    st.write("Upload your CV and find the top matching job descriptions from our dataset.")

    sample_size = st.number_input("Enter the number of jobs to look for (more jobs take longer):", min_value=1, value=100, step=1)
    location = st.text_input("Enter the location for job search (e.g., city, state):")
    position = st.text_input("Enter the position you are looking for:")
    uploaded_cv = st.file_uploader("Choose a CV file", type="pdf")
    
    if uploaded_cv is not None:
        with st.spinner('Processing...'):
            cv_text =extract_text_from_pdf(uploaded_cv)
            jobs = scrape_my_jobs(location=location, position=position, results_wanted=sample_size)

            if len(jobs) > 0:
                st.write(f"Found {len(jobs)} jobs matching your search criteria.")
                descriptions = [description for description in jobs['description']]
                match_results = find_top_matches(cv_text, descriptions, top_n=5)
                print(match_results)
                st.write("Top matching job descriptions:")

                def get_positional_index(index,similarities):
                    for i in range(len(similarities)):
                        if similarities[i][1] == index:
                            return similarities[i][0]
                
                counter = 0
                for score, index  in match_results['overall_similarities']:
                    job_details = jobs.iloc[index]
                    similarities = {
                        'overall_similarity': score,
                        'education_job_similarity': get_positional_index(index,match_results['education_job_similarities']),
                        'experience_job_similarity': get_positional_index(index,match_results['experience_job_similarities']),
                        'education_requirements_similarity': get_positional_index(index,match_results['education_requirements_similarities']),
                        'experience_responsibilities_similarity': get_positional_index(index,match_results['experience_responsibilities_similarities'])

                        # Populate other similarity scores here if available in your data structure
                    }
                    counter += 1
                    display_job_match(counter, job_details, similarities)
            else:
                st.write("No jobs found matching your criteria.")
    
# Page for comparing a CV against a single job posting
def page_single_comparison():
    st.image('image_banner.png', use_column_width=True)
    st.title("CV vs Job Posting Comparison")
    st.write("Upload your CV and a job posting to see if you're a good fit.")
    
    uploaded_cv = st.file_uploader("Choose your CV (PDF)", type="pdf", key="cv_single")
    uploaded_job = st.file_uploader("Choose the job posting (PDF)", type="pdf", key="job_single")
    job_text = st.text_area("Enter the job description here:", height=150, key="job_enhance_text")

    if st.button("Compare"):
            with st.spinner('Processing...'):
                # Read the uploaded CV and job posting files
                if uploaded_cv is not None:
                    cv_text = extract_text_from_pdf(uploaded_cv)
                    if uploaded_job is not None:
                        job_text = extract_text_from_pdf(uploaded_job)

                cv_sections = classify_cv_sections(cv_text)
                cv_education = cv_sections['education']
                cv_experience = cv_sections['experience']

                job_sections = classify_job_description(job_text) 
                job_requirements = job_sections['requirements']
                job_responsibilities = job_sections['responsibilities']


                # Calculate similarity
                overall_sim = calculate_similarity(cv_text, job_text)
                education_requirements_similarities = calculate_similarity(cv_education, job_requirements)
                experience_responsibilities_similarities = calculate_similarity(cv_experience, job_responsibilities)
                
                # Display result
                if overall_sim >= 0.55:
                    st.success(f"You're a good fit! Overall Similarity: {overall_sim * 100:.2f}%)")
                    st.sucess(f" Education & Requirements Similarity: {education_requirements_similarities * 100:.2f}%")
                    st.sucess(f" Experience & Responsibilities Similarity: {experience_responsibilities_similarities * 100:.2f}%")
                else:          
                    st.warning(f"I don't know, maybe try something else? (Similarity: {overall_sim * 100:.2f}%")
                    st.warning(f" Education & Requirements Similarity: {education_requirements_similarities * 100:.2f}%")
                    st.warning(f" Experience & Responsibilities Similarity: {experience_responsibilities_similarities * 100:.2f}%")


# Page for enhancing CV based on job posting
def page_enhance_cv():
    st.image('image_banner.png', use_column_width=True)
    st.title("Enhance my CV")
    st.write("Upload your CV and the job posting to get suggestions on how to improve your CV to match the job posting.")

    uploaded_cv = st.file_uploader("Choose your CV (PDF)", type="pdf", key="cv_enhance")
    uploaded_job = st.file_uploader("Choose the job posting (PDF)", type="pdf", key="job_enhance")
    job_text = st.text_area("Enter the job description here:", height=150, key="job_enhance_text")

    if st.button("Enhance"):
        cv_text = extract_text_from_pdf(uploaded_cv)
        cv_sections = classify_cv_sections(cv_text)
        cv_education = cv_sections['education']
        cv_experience = cv_sections['experience']
        if uploaded_cv is not None and uploaded_job is not None:
            with st.spinner('Processing...'):
                # Read the uploaded CV and job posting files
                job_text = extract_text_from_pdf(uploaded_job)

                job_sections = classify_job_description(job_text) 
                job_requirements = job_sections['requirements']
                job_responsibilities = job_sections['responsibilities']

                missing_keywords_overall= get_missing_keywords(cv_text, job_text)
                missing_keywords_education = get_missing_keywords(cv_education, job_requirements)
                missing_keywords_experience = get_missing_keywords(cv_experience, job_responsibilities)
                
        else:
            with st.spinner('Processing...'):
                job_text = job_text
                job_sections = classify_job_description(job_text) 
                job_requirements = job_sections['requirements']
                job_responsibilities = job_sections['responsibilities']

                missing_keywords_overall = get_missing_keywords(cv_text, job_text)
                missing_keywords_education = get_missing_keywords(cv_education, job_requirements)
                missing_keywords_experience = get_missing_keywords(cv_experience, job_responsibilities)

        
      # Display result
        st.write(f"Suggestions to enhance your CV:")
                
        if missing_keywords_overall:
            st.success(f"Overall Keywords to Add: {', '.join(missing_keywords_overall)}")
        else:
            st.info("Your CV already contains all the important keywords.")
        
        if missing_keywords_education:
            st.success(f"Education Keywords to Add: {', '.join(missing_keywords_education)}")
        else:
            st.info("Your CV already contains all the important education keywords.")
        
        if missing_keywords_experience:
            st.success(f"Experience Keywords to Add: {', '.join(missing_keywords_experience)}")
        else:
            st.info("Your CV already contains all the important experience keywords.")
          

                
        

# Page for generating interview questions based on job description
def page_interview_questions():
    st.image('image_banner.png', use_column_width=True)
    st.title("Interview Question Generator")
    st.write("Upload a job posting to generate potential interview questions based on the job description.")
    

    uploaded_job = st.file_uploader("Choose the job posting (PDF)", type="pdf", key="job_questions")
    open_ai_key = st.text_input("Enter your OpenAI API key:")
    job_text = st.text_area("Enter the job description here:", height=150, key="job_questions_text")
 
    if st.button("Generate Questions"):
        if uploaded_job is not None:
            with st.spinner('Processing...'):
                # Read the uploaded job posting file
                job_text = extract_text_from_pdf(uploaded_job)

                # Generate interview questions
                interview_questions = generate_interview_questions(job_text, open_ai_key)

                # Display result
                st.write("Generated Interview Questions:")
                for question in interview_questions:
                    st.write(f"- {question}")
        else:
            with st.spinner('Processing...'):
            
                    # Generate interview questions
                interview_questions = generate_interview_questions(job_text, open_ai_key)
                # Display result
                st.write("Generated Interview Questions:")
                for question in interview_questions:
                    st.write(f"- {question}")
