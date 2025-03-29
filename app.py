# Polished Flask RAG App with Dynamic Context Filtering
from flask import Flask, request, jsonify
from flask_cors import CORS
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from openai import OpenAI
import os
import json
import base64
import zipfile
from fpdf import FPDF
from docx import Document
from datetime import datetime
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

client = OpenAI()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL")
USAGE_FILE = "usage_log.json"

# Load FAISS index and document chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("index.faiss")
docs = []
for i in range(1, 5):
    with open(f"docs_part{i}.pkl", "rb") as f:
        docs.extend(pickle.load(f))
print(f"✅ Loaded {len(docs)} total chunks")

# Usage tracking
def load_usage():
    if not os.path.exists(USAGE_FILE):
        return {}
    with open(USAGE_FILE, 'r') as f:
        return json.load(f)

def save_usage(data):
    with open(USAGE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def can_use(email):
    usage = load_usage()
    return usage.get(email, 0) < 5

def record_use(email):
    usage = load_usage()
    usage[email] = usage.get(email, 0) + 1
    save_usage(usage)

# Main route
@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    name = data.get("full_name")
    email = data.get("email")
    query = data.get("query")
    job_title = data.get("job_title", "")
    discipline = data.get("discipline", "")
    search_type = data.get("search_type", "General")

    if not name or not email or not query:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    if not can_use(email):
        return jsonify({"status": "error", "message": "limit_reached"}), 403

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Vector search
    query_vector = model.encode([query])[0].astype("float32").reshape(1, -1)
    D, I = index.search(query_vector, k=10)
    raw_chunks = [docs[i]['text'] for i in I[0] if i < len(docs)]

    # Dynamic keyword filtering
    keyword_map = {
        "Health_Safety": ["accident", "electrical", "injury", "incident", "report", "HSE", "RIDDOR"],
        "Fire_Safety": ["fire", "evacuation", "drill", "alarm", "extinguisher", "flammable"],
        "Training": ["training", "instruction", "briefing", "session", "competency"],
        "General": []
    }
    important_terms = keyword_map.get(search_type, [])

    context_chunks = [
        chunk for chunk in raw_chunks
        if len(chunk.strip().split()) > 50
        and "glossary" not in chunk.lower()
        and "index" not in chunk.lower()
        and (not important_terms or any(term in chunk.lower() for term in important_terms))
    ][:3]
    context = "\n\n".join(context_chunks)

    # Role-aware prompt
    if job_title.lower() in ["director", "trustee"]:
        tone = f"""
You are a UK-based expert assistant. The following question may relate to a workplace accident or health and safety incident.

Your response must be comprehensive and include:
- Emergency actions (first aid, isolating the area, alerting authorities)
- Legal duties under UK law, including RIDDOR reporting
- Notification requirements (HSE, employer, local authority)
- Follow-up and internal investigation steps
- Preventive measures for future incidents

Structure your response using clear headings and bullet points.
Use a professional but readable tone appropriate for a {job_title} in the {discipline} sector.

End with: 'Would you like assistance drafting a RIDDOR submission or incident report template?'
"""
    elif job_title.lower() in ["site supervisor", "foreman"]:
        tone = f"""
You are a UK-based assistant. If the query involves an H&S incident, outline the legally required actions under UK regulations.
Respond with practical steps relevant to the supervisor's responsibilities.
"""
    else:
        tone = f"""
You are a UK-based assistant. Answer clearly and simply. If the query is about an H&S incident, mention whether it is reportable and who should be informed.
"""

    prompt = f"""
{tone}

Context:
{context}

Question:
{query}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return jsonify({"status": "error", "message": f"OpenAI error: {str(e)}"}), 500

    disclaimer = (
        "DISCLAIMER: This response was generated by AI based on available UK-specific documentation. "
        "While care has been taken to ensure relevance, the information provided is for general guidance only "
        "and does not constitute legal or professional advice. Please consult a qualified expert before acting on any recommendation."
    )

    full_text = f"""Query: {query}

Response (generated {timestamp}):

{answer}

{disclaimer}
"""

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, full_text.encode('latin-1', 'replace').decode('latin-1'))
        pdf_file = "response.pdf"
        pdf.output(pdf_file)

        doc = Document()
        doc.add_paragraph(full_text)
        docx_file = "response.docx"
        doc.save(docx_file)

        context_text = f"""Query: {query}

Context Chunks Used (generated {timestamp}):

{context}

{disclaimer}
"""
        context_pdf = FPDF()
        context_pdf.add_page()
        context_pdf.set_font("Arial", size=12)
        context_pdf.multi_cell(0, 10, context_text.encode('latin-1', 'replace').decode('latin-1'))
        context_file = "context.pdf"
        context_pdf.output(context_file)

        zip_file = "response.zip"
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            zipf.write(pdf_file)
            zipf.write(docx_file)
            zipf.write(context_file)

        with open(zip_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        attachment = Attachment(
            FileContent(encoded),
            FileName("response.zip"),
            FileType("application/zip"),
            Disposition("attachment")
        )

        message = Mail(
            from_email=FROM_EMAIL,
            to_emails=email,
            subject="Your AI Response",
            plain_text_content="Attached is your AI-generated response."
        )
        message.attachment = attachment

        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)

    except Exception as e:
        return jsonify({"status": "error", "message": f"Email error: {str(e)}"}), 500

    record_use(email)
    return jsonify({"status": "success", "message": "Response emailed successfully."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)
