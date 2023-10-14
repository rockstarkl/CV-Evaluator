# importing the useful packages
import csv
import os
import nltk
import spacy as spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import shutil
from textblob import TextBlob
# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_md")

### Function to extract the text from the PDF files
def extract_text_from_pdf(pdf_file_path):
    text = ''
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text
### Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing, stopword removal, and lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if
                    token.isalnum() and token.lower() not in stopwords.words('english')]

    return ' '.join(clean_tokens)



### Feature extraction function
def extract_features(texts):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix


### Similarity score calculation function
def calculate_similarity(tfidf_matrix):
    # Assuming tfidf_matrix contains two rows (one for CV and one for job description)
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score

def convert_to_percentage(similarity_score):
    # Map the similarity score from [-1, 1] to [0%, 100%]
    percentage = (similarity_score + 1) * 50
    return percentage

def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Determine overall sentiment polarity (positive, negative, or neutral)
    sentiment = analysis.sentiment.polarity
    return sentiment

### Function to perform keyword extraction and highlight relevant skills and experiences
def extract_and_highlight_keywords(cv_text, job_description_text):
    # Tokenize and preprocess CV and job description
    preprocessed_cv_text = preprocess_text(cv_text)
    preprocessed_job_description_text = preprocess_text(job_description_text)

    # Process text using spaCy for NER and keyword extraction
    cv_doc = nlp(preprocessed_cv_text)
    job_description_doc = nlp(preprocessed_job_description_text)

    # Extract entities (e.g., skills, experiences) from the CV
    cv_entities = set([ent.text for ent in cv_doc.ents if ent.label_ in ["ORG", "GPE", "PERSON"]])

    # Extract entities (e.g., skills, experiences) from the job description
    job_description_entities = set([ent.text for ent in job_description_doc.ents if ent.label_ in ["ORG", "GPE", "PERSON"]])

    # Find common entities (matching skills or experiences)
    common_entities = cv_entities.intersection(job_description_entities)

    # Highlight matching entities in the CV text
    highlighted_cv_text = preprocessed_cv_text
    for entity in common_entities:
        highlighted_cv_text = highlighted_cv_text.replace(entity, f"**{entity}**")

    return highlighted_cv_text


### sentiment Category
def categorize_sentiment(sentiment_score):
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

### getting the folder location and CV's
cwd = os.getcwd()
input_folder = os.path.join(cwd, "input")
out_folder = os.path.join(cwd, "output")
job_desscription = os.path.join(cwd,"job_desc")

pass_folder = os.path.join(out_folder, "Pass")
fail_folder = os.path.join(out_folder, "Fail")
os.makedirs(pass_folder, exist_ok=True)
os.makedirs(fail_folder, exist_ok=True)

### Reading the Job description
File_Job_desc = 'JD_TA.pdf' # Name of the Description file
Job_description_text = extract_text_from_pdf(job_desscription+'//'+File_Job_desc)

preprocessed_job_description_text = preprocess_text(Job_description_text)

cv_score = []
### Loop to read mutiple files CV
list_of_files = os.listdir(input_folder)
for file in list_of_files:
        cv_text = extract_text_from_pdf(input_folder+'//'+file)
        print(file)
        preprocessed_cv_text = preprocess_text(cv_text)
        # Extract features for both CV and job description
        tfidf_matrix = extract_features([preprocessed_cv_text, preprocessed_job_description_text])
        # Calculate similarity score
        similarity_score = calculate_similarity(tfidf_matrix)
        print(f"Similarity Score: {similarity_score}")
        # Method to convert into Percentage
        percentage_similarity = convert_to_percentage(similarity_score)
        print(f"Similarity Percentage: {percentage_similarity}")

        # Perform Sentiment analysis
        sentiment = analyze_sentiment(preprocessed_cv_text)
        # Collecting all files output in cv_score
        sentiment_category = categorize_sentiment(sentiment)
        cv_score.append((file, percentage_similarity, sentiment_category))
        # Perform keyword extraction and highlighting
        highlighted_cv_text = extract_and_highlight_keywords(cv_text, Job_description_text)
        cv_score.append((file, percentage_similarity, sentiment_category,highlighted_cv_text))
        print(highlighted_cv_text)
        #score.append(percentage_similarity)
        if percentage_similarity < 70:
            print("Fail")
            shutil.move(os.path.join(input_folder, file), os.path.join(fail_folder, file))
        else:
            print("pass")
            shutil.move(os.path.join(input_folder, file), os.path.join(pass_folder, file))
        print('##############################################################')
### reverse function for Rank
cv_score.sort(key=lambda x: x[1], reverse=True)

print(cv_score)

# Write CV file names and scores to a CSV file
csv_file_path = os.path.join(out_folder, "Evaluation_result.csv")
with open(csv_file_path, mode='w', newline='') as csv_file:
    fieldnames = ['CV File Name', 'Similarity Score (%)','Result','Sentiment','Highli_text']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for cv_file, score,sentiment_category,highlighted_cv_text in cv_score:
        if score < 70:
            ot = "Fail"
            writer.writerow({'CV File Name': cv_file, 'Similarity Score (%)': score, 'Result': ot, 'Sentiment':sentiment_category, 'Highli_text':highlighted_cv_text})
        else:
            ot = "Pass"
            writer.writerow({'CV File Name': cv_file, 'Similarity Score (%)': score, 'Result': ot, 'Sentiment':sentiment_category, 'Highli_text':highlighted_cv_text})