from flask import Flask, render_template, request, jsonify, send_file, Response
import requests
import pdfplumber
import pandas as pd
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

chat_history = []
last_uploaded_df = None  # <-- Stores the latest uploaded DataFrame

# === LLaMA Query: non-streaming ===
def query_llama(message, history):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama2",
        "messages": history + [{"role": "user", "content": message}],
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['message']['content']
    except requests.exceptions.RequestException as e:
        return f"⚠️ LLaMA 2 error: {e}"

# === LLaMA Stream Response ===
def stream_llama_response(message, history):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama2",
        "messages": history + [{"role": "user", "content": message}],
        "stream": True
    }

    def generate():
        with requests.post(url, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'message' in chunk and 'content' in chunk['message']:
                            yield chunk['message']['content']
                    except Exception as e:
                        yield f"\n[stream error: {e}]"

    return Response(generate(), content_type='text/plain')

# === PDF Text Extraction ===
def extract_text_from_pdf(pdf_path):
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
    except Exception as e:
        return f"⚠️ Error reading PDF: {e}"
    return full_text.strip()

# === Homepage ===
@app.route('/')
def index():
    return render_template('chat.html')

# === Chat Stream Endpoint ===
@app.route('/stream', methods=['POST'])
def stream():
    user_input = request.form['message']
    chat_history.append({"role": "user", "content": user_input})

    if last_uploaded_df is not None:
        preview = last_uploaded_df.head(10).to_markdown(index=False)
        columns = ', '.join(last_uploaded_df.columns)
        rowcount = len(last_uploaded_df)

        prompt = f"""You are a data analyst working with the following dataset.

The dataset contains {rowcount} rows with columns: {columns}

Here are the first few rows:
{preview}

User's question: {user_input}

Answer clearly using insights from the dataset. Do not output code or repeat the data table.
"""
        return stream_llama_response(prompt, [])
    
    return stream_llama_response(user_input, chat_history)

# === One-shot send (not used with stream) ===
@app.route('/send', methods=['POST'])
def send():
    user_input = request.form['message']
    chat_history.append({"role": "user", "content": user_input})
    bot_reply = query_llama(user_input, chat_history)
    chat_history.append({"role": "assistant", "content": bot_reply})
    return jsonify({'reply': bot_reply})

# === PDF Upload & Summarize ===
@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({'reply': "⚠️ No PDF file uploaded."})

        file = request.files['pdf']
        if file.filename == '':
            return jsonify({'reply': "⚠️ File name is empty."})

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        extracted_text = extract_text_from_pdf(filepath)
        if not extracted_text or extracted_text.startswith("⚠️"):
            return jsonify({'reply': extracted_text or "⚠️ PDF is unreadable or empty."})

        summary_prompt = f"Please summarize the following document:\n\n{extracted_text[:3000]}"
        bot_reply = query_llama(summary_prompt, [])

        summary_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sumrpt.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(bot_reply)

        return jsonify({'reply': bot_reply, 'download_url': '/download-summary'})

    except Exception as e:
        return jsonify({'reply': f"⚠️ Unexpected error: {str(e)}"})

# === PDF Summary Download ===
@app.route('/download-summary')
def download_summary():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'sumrpt.txt')
    return send_file(path, as_attachment=True) if os.path.exists(path) else ("Summary not found", 404)

# === CSV/Excel Upload & Data Analysis ===
@app.route('/upload-data', methods=['POST'])
def upload_data():
    global last_uploaded_df
    try:
        file = request.files.get('datafile')
        if not file or file.filename == '':
            return jsonify({'reply': "⚠️ No data file uploaded."})

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            return jsonify({'reply': "⚠️ Only CSV or Excel files are supported."})

        last_uploaded_df = df.copy()

        preview = df.head(10).to_markdown(index=False)
        prompt = f"""You are a data analyst. Based on this dataset, provide key insights.

Here are the first few rows:
{preview}

Avoid code or repeating the raw data.
"""

        reply = query_llama(prompt, [])

        insights_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data_insights.txt')
        with open(insights_path, 'w', encoding='utf-8') as f:
            f.write(reply)

        return jsonify({'reply': reply, 'download_url': '/download-data-insights'})

    except Exception as e:
        return jsonify({'reply': f"⚠️ Error analyzing data: {str(e)}"})

# === Insights Download ===
@app.route('/download-data-insights')
def download_data_insights():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'data_insights.txt')
    return send_file(path, as_attachment=True) if os.path.exists(path) else ("Insights not found", 404)

# === Start Server ===
if __name__ == '__main__':
    app.run(debug=True)
