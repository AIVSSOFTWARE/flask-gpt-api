# ✅ Flask app with RAG using FAISS + sentence-transformers + multi-part docs
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

# ✅ Load RAG components from split docs
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("index.faiss")
docs = []
for i in range(1, 4):  # Adjust if you have more parts
    part_file = f"docs_part{i}.pkl"
    if os.path.exists(part_file):
        with open(part_file, "rb") as f:
            part = pickle.load(f)
            docs.extend(part)
print(f"✅ Loaded {len(docs)} total chunks")

# ---------------------------
# Usage Tracking
# ---------------------------
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

# ---------------------------
# Main Endpoint
# ---------------------------
@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    name = data.get("full_name")
    email = data.get("email")
    query = data.get("query")

    if not name or not email or not query:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    if not can_use(email):
        return jsonify({"status": "error", "message": "limit_reached"}), 403

    # ✅ RAG: Embed query and search FAISS index
    query_vector = model.encode([query])[0].astype("float32").reshape(1, -1)
    D, I = index.search(query_vector, k=5)
    context_chunks = [docs[i]['text'] for i in I[0] if i < len(docs)]
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an assistant. Use the context below to answer the question.

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

    disclaimer = "DISCLAIMER: This response is AI-generated and may contain inaccuracies."
    full_text = f"Query: {query}\n\nContext:\n{context}\n\nResponse:\n{answer}\n\n{disclaimer}"

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, full_text)
    pdf_file = "response.pdf"
    pdf.output(pdf_file)

    # Generate DOCX
    docx_file = "response.docx"
    doc = Document()
    doc.add_paragraph(full_text)
    doc.save(docx_file)

    # Create ZIP
    zip_file = "response.zip"
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(pdf_file)
        zipf.write(docx_file)

    try:
        with open(zip_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        attachment = Attachment(
            FileContent(encoded),
            FileName(zip_file),
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

# ---------------------------
# Run Locally
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)