from flask import Flask, request, jsonify
import PyPDF2
import io
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

app = Flask(__name__)

# تحميل موارد NLTK مرة واحدة
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

stop_words_english = set(stopwords.words("english"))

def split_by_headings(text):
    """يقسم النص إلى أقسام بناءً على رؤوس: Skills, Experience, Education"""
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

@app.route('/match', methods=['POST'])
def match_resumes():
    # استلام JSON الوظائف والمهارات
    job_requirements = None
    if request.is_json:
        job_requirements = request.get_json().get('job_requirements', None)
    else:
        job_requirements = request.form.get('job_requirements', None)
        if job_requirements:
            import json
            job_requirements = json.loads(job_requirements)
    if not job_requirements:
        return jsonify({"error": "Missing job_requirements JSON"}), 400

    # استلام ملفات السير الذاتية
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

        # تقسيم النص إلى أقسام واستخراج قسم المهارات
        sections, sections_inx = split_by_headings(text)
        skills_section = ""
        for i, idx in enumerate(sections_inx):
            if idx == 0:  # قسم المهارات
                skills_section = sections[i]
                break
        text_to_process = skills_section if skills_section else text

        processed_text = preprocess_text(text_to_process)

        # حساب التطابق لكل وظيفة
        job_scores = {}
        best_match = None
        best_score = -1
        for job_title, skills in job_requirements.items():
            score = calculate_match_percentage(processed_text, skills)
            job_scores[job_title] = score
            if score > best_score:
                best_score = score
                best_match = job_title

        # إذا كانت جميع النسب صفر، نعتبر أنه لا توجد وظيفة مناسبة
        if best_score == 0:
            best_match = "No suitable job found"

        results.append({
            "filename": file.filename,
            "best_job_match": best_match,
            "best_match_percentage": best_score,
            "all_matches": job_scores
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
