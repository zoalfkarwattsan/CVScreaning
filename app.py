import nltk
from flask import Flask, request, jsonify
import PyPDF2
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import json

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
import numpy as np
import io

app = Flask(__name__)
svm_model = joblib.load('svm_model.pkl')
pca = joblib.load('pca_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
sentence_model = SentenceTransformer('all-mpnet-base-v2')


def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_stream.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None


def predict_category(text):

    sentence_emb = sentence_model.encode([text])


    reduced_features = pca.transform(sentence_emb)


    predicted_label = svm_model.predict(reduced_features)[0]
    return label_encoder.inverse_transform([predicted_label])[0]


###################################################

puncuation = set(string.punctuation)
stop_words_english = set(stopwords.words("english"))


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)


def preprocess_cv_text(text):
    sentences = sent_tokenize(text)  # استخدم النص الأصلي لتقسيم الجمل
    collected_words = []

    keywords = ['skills', 'education']

    for sent in sentences:
        sent_lower = sent.lower()
        if any(keyword in sent_lower for keyword in keywords):

            clean_sent = re.sub('[^a-zA-Z]', ' ', sent_lower)
            words = word_tokenize(clean_sent)
            words = [word for word in words if word not in stop_words_english]
            tagged_words = pos_tag(words)
            filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
            collected_words.extend(filtered_words)

    return ' '.join(collected_words)

def preprocess_job_text(text):
    sentences = sent_tokenize(text)  # Tokenize the text into sentences
    collected_words = []

    for sent in sentences:
        sent_lower = sent.lower()  # Convert the sentence to lowercase
        # Clean the sentence (remove non-alphabetic characters)
        clean_sent = re.sub('[^a-zA-Z]', ' ', sent_lower)
        words = word_tokenize(clean_sent)  # Tokenize the cleaned sentence
        words = [word for word in words if word not in stop_words_english]  # Remove stopwords
        tagged_words = pos_tag(words)  # Part of speech tagging
        filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]  # Remove unwanted POS tags
        collected_words.extend(filtered_words)  # Collect the filtered words

    return ' '.join(collected_words)



def get_embeddings(text):
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings


def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


@app.route('/recommend_cvs', methods=['POST'])
def match_resumes():

    if 'position' not in request.form:
        return jsonify({"error": "Missing 'position' field"}), 400
    if 'job_description_pdf' not in request.files:
        return jsonify({"error": "Missing job description PDF file"}), 400
    if 'resumes' not in request.files:
        return jsonify({"error": "Missing resumes files"}), 400

    position = request.form['position']
    job_desc_file = request.files['job_description_pdf']
    resume_files = request.files.getlist('resumes')


    job_desc_text = extract_text_from_pdf(job_desc_file.stream)
    if not job_desc_text:
        return jsonify({"error": "Failed to extract text from job description PDF"}), 400


    job_desc_features = preprocess_job_text(job_desc_text)

    job_desc_category = predict_category(job_desc_text)

    job_desc_emb = get_embeddings(job_desc_features)

    matching_resumes = []
    non_matching_resumes = []

    for file in resume_files:
        text = extract_text_from_pdf(file.stream)
        if not text:
            continue
        features = preprocess_cv_text(text)
        category = predict_category(text)

        emb = get_embeddings(features)

        resume_info = {
            'filename': file.filename,
            'text': text,
            'category': category,
            'embedding': emb
        }

        if category == job_desc_category:
            matching_resumes.append(resume_info)
        else:
            non_matching_resumes.append({
                'filename': file.filename,
                'category': category
            })

    if not matching_resumes:
        return jsonify({"error": "No resumes match the job description category"}), 404


    embeddings = np.vstack([r['embedding'] for r in matching_resumes])
    similarities = cosine_similarity(job_desc_emb, embeddings)[0]

    for i, sim in enumerate(similarities):
        matching_resumes[i]['similarity_score'] = float(sim)

    matching_resumes.sort(key=lambda x: x['similarity_score'], reverse=True)

    top_3_matches = matching_resumes[:3]

    other_matches = matching_resumes[3:]

    return jsonify({
        "position": position,
        "job_description_category": job_desc_category,
        "top_3_matches": [
            {
                "filename": r['filename'],
                "similarity_score": r['similarity_score'],
                "category": r['category']
            } for r in top_3_matches
        ],
        "other_matching_resumes": [
            {
                "filename": r['filename'],
                "similarity_score": r['similarity_score'],
                "category": r['category']
            } for r in other_matches
        ],
        "non_matching_resumes": non_matching_resumes
    })



def split_by_headings(text):

    sections = []
    sections_inx = []
    current_section = ""
    now = 0
    for line in text.splitlines():
        if "Skill" in line or "Experience" in line or "Education" in line:
            if current_section or now == 0:
                if "Skill" in line:
                    sections_inx.append(0)
                elif "Experience" in line:
                    sections_inx.append(1)
                elif "Education" in line:
                    sections_inx.append(2)
            if current_section:
                sections.append(current_section)
            current_section = line
            now = 1
        else:
            if now:
                current_section += " " + line
    if current_section:
        sections.append(current_section)

    start = 100
    skill_text = ""
    for ii in range(len(sections)):
        if sections_inx[ii] == 0:
            skill_text += sections[ii]
            if start == 100:
                start = ii
        if start < 100:
            sections[start] = skill_text
    return sections, sections_inx

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z ]', ' ', text)
    sentences = sent_tokenize(text)
    collected_words = []
    for sent in sentences:
        clean_sent = re.sub('[^a-zA-Z ]', ' ', sent)
        words = word_tokenize(clean_sent)
        words = [word for word in words if word not in stop_words_english]
        tagged_words = pos_tag(words)
        filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
        collected_words.extend(filtered_words)
    return ' '.join(collected_words)

def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_stream.read()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def calculate_match_percentage(text, keywords):
    text_words = set(text.lower().split())
    keywords_set = set([kw.lower() for kw in keywords])
    matched = keywords_set.intersection(text_words)
    if len(keywords_set) == 0:
        return 0
    return int(len(matched) / len(keywords_set) * 100)


try:
    with open('jobs_data.json', 'r', encoding='utf-8') as f:
        job_data = json.load(f)
except FileNotFoundError:
    print("Error: jobs_data.json not found")
    job_data = {}

@app.route('/recommend_jobs', methods=['POST'])
def match_jobs():
    if 'resumes' not in request.files:
        return jsonify({"error": "Missing resume files"}), 400
    resume_files = request.files.getlist('resumes')
    if not resume_files:
        return jsonify({"error": "No resume files uploaded"}), 400

    results = []
    for file in resume_files:
        text = extract_text_from_pdf(file.stream)
        if not text:
            results.append({"filename": file.filename, "error": "Could not extract text"})
            continue

        sections, sections_inx = split_by_headings(text)
        skills_section = ""
        for i, idx in enumerate(sections_inx):
            if idx == 0:
                skills_section = sections[i]
                break
        text_to_process = skills_section if skills_section else text
        predicted_category = predict_category(text)
        processed_text = preprocess_text(text_to_process)




        if predicted_category not in job_data:
            results.append({
                "filename": file.filename,
                "error": f"Category '{predicted_category}' not found in job data"
            })
            continue

        job_scores = {}
        best_match = None
        best_match_percentage = -1

        for job in job_data[predicted_category]:
            job_title = job['Job Title']
            skills = job['Key Skills']
            score = calculate_match_percentage(processed_text, skills)
            job_scores[job_title] = score
            if score > best_match_percentage:
                best_match_percentage = score
                best_match = job_title

        if best_match_percentage == 0:
            best_match = "No suitable job found"

        results.append({
            "filename": file.filename,
            "predicted_category": predicted_category,
            "best_job_match": best_match,
            "best_match_percentage": best_match_percentage,
            "all_matches_in_category": job_scores
        })

    return jsonify(results)
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
