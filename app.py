
import os
import faiss
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from fpdf import FPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import base64
import zipfile
from datetime import datetime
import openai

app = Flask(__name__)
CORS(app)

# Load model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
openai.api_key = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL")

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()

    name = data.get("full_name")
    email = data.get("email")
    query = data.get("query")
    job_title = data.get("job_title")
    discipline = data.get("discipline", "").lower().replace(" ", "_")
    search_type = data.get("search_type")
    site = data.get("site")
    supervisor_required = data.get("supervisor_required")
    supervisor_email = data.get("supervisor_email")
    supervisor_name = data.get("supervisor_name")
    funnel_1 = data.get("funnel_1")
    funnel_2 = data.get("funnel_2")
    funnel_3 = data.get("funnel_3")
    dropdown_bonus = data.get("dropdown_bonus")

    # Determine authority level
    job_rank_map = {
        "Managing Director": 10,
        "CEO": 10,
        "Board Member": 9,
        "Strategy Consultant": 8,
        "Senior Advisor": 8,
        "Operations Lead": 6,
        "Project Manager": 5,
        "HR Executive": 5,
        "Analyst": 3,
        "Coordinator": 3,
        "Intern": 1
    }
    rank = job_rank_map.get(job_title, 5)

    # Load index and docs
    try:
        index = faiss.read_index(f"indexes/{discipline}_index.faiss")
        with open(f"indexes/{discipline}_docs.pkl", "rb") as f:
            docs = pickle.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": f"FAISS load error: {str(e)}"}), 500

    # Embed query and search
    query_vector = embed_model.encode([query])
    D, I = index.search(query_vector, k=10)

    # Collect top chunks
    context_chunks = [
        docs[i]['text'] for i in I[0]
        if i < len(docs) and len(docs[i]['text'].strip()) > 200
    ][:3]
    context = "\n\n".join(context_chunks)

    # Compose GPT prompt
    prompt = f"""You are an expert advisor responding to a strategic management query.

Tone: Professional and UK-specific.
Discipline: {discipline.title()}
Job Title: {job_title} (Level {rank})
Search Type: {search_type}
Urgency: {dropdown_bonus}
Site: {site}

Context:
{context}

Question:
{query}
"""

    # Get GPT response
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response['choices'][0]['message']['content']
    except Exception as e:
        return jsonify({"status": "error", "message": f"OpenAI error: {str(e)}"}), 500

    # Format full text for docs
    disclaimer = "DISCLAIMER: This AI-generated response may include inaccuracies. Review before acting. AIVS Software Limited (c) 2025"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    full_text = f"Query by: {name} ({email})\nDate: {timestamp}\n\n{query}\n\n---\n\nResponse:\n{answer}\n\n---\n\n{disclaimer}"

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, full_text.encode('latin-1', 'replace').decode('latin-1'))
    pdf_file = "response.pdf"
    pdf.output(pdf_file)

    # Create DOCX
    doc = Document()
    doc.add_paragraph(full_text)
    docx_file = "response.docx"
    doc.save(docx_file)

    # Context PDF
    context_text = f"Query: {query}\n\nContext Chunks Used:\n\n{context}\nDate: {timestamp}\n\n{disclaimer}"
    context_pdf = FPDF()
    context_pdf.add_page()
    context_pdf.set_font("Arial", size=12)
    context_pdf.multi_cell(0, 10, context_text.encode('latin-1', 'replace').decode('latin-1'))
    context_file = "context.pdf"
    context_pdf.output(context_file)

    # Zip all files
    zip_file = "response.zip"
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(pdf_file)
        zipf.write(docx_file)
        zipf.write(context_file)

    # Email to recipient
    try:
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
            subject="Your Strategic Management AI Response",
            plain_text_content="Please find your response attached."
        )
        message.attachment = attachment

        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Email error: {str(e)}"}), 500

    return jsonify({"status": "success", "message": "Response emailed successfully."})

