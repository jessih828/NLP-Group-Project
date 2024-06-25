# Rephraise: Resume & Job Description Matcher 

Rephraise was created to address the challenges of finding a job in a competitive market. In 2024, the job market is particularly tough, and it's crucial to enhance our chances by improving our CVs and speeding up the job search process. By leveraging NLP, we want to analyzes and matches CVs with job descriptions, ensuring that job seekers present their qualifications in the best possible light.

## Core Functionalities

### 1. Text Extraction
- **PDF to Text Conversion:** Uses `pdfplumber` to convert PDF documents into text, facilitating the processing of both CVs and job descriptions.

### 2. CV and Job Description Analysis
- **Text Classification:** Employs `SpaCy` for named entity recognition (NER) and to classify sections of CVs and job descriptions into categories such as education and experience.
- **Keyword Extraction:** Utilizes `KeyBERT` to identify and suggest important keywords missing from a CV compared to a job description.

### 3. Text Embedding and Similarity Calculation
- **Text Embedding:** Leverages the `Sentence-BERT` model to embed textual data.
- **Similarity Calculation:** Computes cosine similarities to identify the top job descriptions that match a user's CV.

### 4. Interactive Features
- **Job Matching:** Compares a user's CV against multiple job descriptions to find the best matches, presenting detailed similarity scores.
- **CV Enhancement:** Offers actionable suggestions on how to refine a CV based on the requirements and responsibilities outlined in a job description.
- **Interview Preparation:** Generates interview questions tailored to specific job descriptions using OpenAI's `GPT-3.5` model.

## Additional Features
- **Real-Time Job Listings:** Integrates scraping capabilities to fetch real-time job listings from platforms like LinkedIn, further supporting job seekers in their search for relevant positions.

## How to Use

1. **Go to the `test.ipynb` notebook:**
   - Open the `test.ipynb` file in your preferred Jupyter Notebook environment.

2. **Run the cells:**
   - Execute the cells sequentially to see how Rephraiser processes and analyzes your CV against job descriptions.