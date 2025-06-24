import sqlite3
import pandas as pd
import re
from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nltk
from nltk.corpus import stopwords
from datetime import datetime, timedelta
import json
import random
import uuid
import os
from werkzeug.utils import secure_filename
import requests
import zipfile
import io

# Suppress TensorFlow oneDNN messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set NLTK data path
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.expanduser('~'), 'Downloads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize SQLite databases
def init_db():
    conn = sqlite3.connect('phone_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS phone_data
                (phone_number TEXT PRIMARY KEY, owner_name TEXT, location TEXT,
                spam_status TEXT, active_time TEXT, message_type TEXT,
                spam_score INTEGER, spam_reports INTEGER)''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_phone_number ON phone_data(phone_number)')
    conn.commit()
    conn.close()
    
    conn = sqlite3.connect('security_logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS security_logs
                (phone_number TEXT, event_time TEXT, event_type TEXT, description TEXT)''')
    conn.commit()
    conn.close()
    
    conn = sqlite3.connect('call_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS call_history
                (phone_number TEXT, call_time TEXT, call_type TEXT, contact_number TEXT, duration INTEGER)''')
    conn.commit()
    conn.close()
    
    conn = sqlite3.connect('document_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS document_data
                (id TEXT PRIMARY KEY, phone_number TEXT, name TEXT, address TEXT, details TEXT)''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_phone_number_doc ON document_data(phone_number)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_name_doc ON document_data(name)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_address_doc ON document_data(address)')
    conn.commit()
    conn.close()

# Download and extract SMS Spam Collection Dataset
def download_sms_spam_dataset():
    dataset_path = os.path.join(os.path.expanduser('~'), 'sms_spam_collection')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(dataset_path)
    return os.path.join(dataset_path, 'SMSSpamCollection')

# Populate dummy data (Truecaller-like database)
def populate_dummy_data():
    conn = sqlite3.connect('phone_data.db')
    c = conn.cursor()
    names = ['Ravi Kumar', 'Priya Sharma', 'Anil Singh', 'Deepa Nair', 'Vikram Patel', 'Meena Reddy', 'Arjun Menon', 'Sneha Gupta', 'Rahul Verma', 'Suman Iyer', 'Christopher Berry']
    cities = ['Chennai', 'Delhi', 'Bangalore', 'Mumbai', 'Hyderabad', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Chandigarh']
    prefixes = ['98', '93', '89', '87', '78', '70', '91', '82', '67', '74']
    
    dummy_data = []
    # Include test_data.csv numbers
    dummy_data.append(('9845123760', 'Christopher Berry', 'Bangalore', 'Not Spam', datetime.now().strftime('%H:%M'), 'Personal', 0, 0))
    dummy_data.append(('8210625365', 'Rahul Verma', 'Delhi', 'Not Spam', datetime.now().strftime('%H:%M'), 'Personal', 0, 0))
    
    # Generate 10,000 random numbers
    for i in range(10000):
        phone = f"{random.choice(prefixes)}{random.randint(10000000, 99999999)}"
        name = random.choice(names)
        city = random.choice(cities)
        spam_status = 'Spam' if random.random() < 0.2 else 'Not Spam'
        spam_score = random.randint(60, 95) if spam_status == 'Spam' else random.randint(0, 20)
        spam_reports = random.randint(10, 200) if spam_status == 'Spam' else random.randint(0, 5)
        dummy_data.append((phone, name, city, spam_status, datetime.now().strftime('%H:%M'), 'Personal', spam_score, spam_reports))
    
    c.executemany('INSERT OR IGNORE INTO phone_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)', dummy_data)
    conn.commit()
    conn.close()

# Simulate security logs
def populate_security_logs():
    conn = sqlite3.connect('security_logs.db')
    c = conn.cursor()
    dummy_logs = [
        ('9876543210', '2025-05-06 10:00:00', 'Login Attempt', 'Normal login'),
        ('8765432109', '2025-05-05 15:30:00', 'App Install', 'Suspicious app install'),
        ('8210625365', '2025-05-04 08:45:00', 'Network Access', 'Normal activity')
    ]
    c.executemany('INSERT OR IGNORE INTO security_logs VALUES (?, ?, ?, ?)', dummy_logs)
    conn.commit()
    conn.close()

# Simulate call history
def populate_call_history():
    conn = sqlite3.connect('call_history.db')
    c = conn.cursor()
    dummy_calls = [
        ('9876543210', '2025-05-06 14:20:00', 'Outgoing', '1234567890', 120),
        ('8765432109', '2025-05-05 09:10:00', 'Incoming', '2345678901', 300),
        ('8210625365', '2025-05-04 18:45:00', 'Missed', '3456789012', 0)
    ]
    c.executemany('INSERT OR IGNORE INTO call_history VALUES (?, ?, ?, ?, ?)', dummy_calls)
    conn.commit()
    conn.close()

# Insert CSV or Excel data into document_data.db and phone_data.db
def insert_csv_data(file):
    try:
        filename = secure_filename(file.filename)
        if filename.endswith('.csv'):
            encodings = ['utf-8', 'latin1', 'utf-16']
            delimiters = [',', ';', '\t']
            df = None
            for enc in encodings:
                for delim in delimiters:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, encoding=enc, sep=delim, engine='python')
                        if len(df.columns) >= 4:
                            break
                    except Exception as e:
                        print(f"Encoding {enc} delimiter {delim}: {str(e)}")
                if df is not None:
                    break
            if df is None:
                return {'status': 'Error', 'message': 'Could not read CSV file'}
        elif filename.endswith('.xlsx'):
            file.seek(0)
            df = pd.read_excel(file)
        else:
            return {'status': 'Error', 'message': 'File must be .csv or .xlsx'}
        
        df.columns = [col.lower().strip() for col in df.columns]
        headers = set(df.columns)
        required_headers = {'phone_number', 'name', 'address', 'details'}
        
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame head:", df.head().to_dict(orient='records'))
        
        if not headers.issuperset(required_headers):
            return {'status': 'Error', 'message': f'File must contain {", ".join(required_headers)} columns. Found: {", ".join(headers)}'}
        
        conn_doc = sqlite3.connect('document_data.db', isolation_level=None)
        c_doc = conn_doc.cursor()
        inserted_doc_rows = 0
        
        conn_phone = sqlite3.connect('phone_data.db', isolation_level=None)
        c_phone = conn_phone.cursor()
        inserted_phone_rows = 0
        
        c_doc.execute('BEGIN TRANSACTION')
        c_phone.execute('BEGIN TRANSACTION')
        
        for index, row in df.iterrows():
            try:
                phone_number = str(row['phone_number']).strip()
                name = str(row['name']).strip()
                address = str(row['address']).strip()
                details = str(row['details']).strip()
                if not phone_number or not name:
                    print(f"Row {index} skipped: Phone number or name is empty")
                    continue
                
                c_doc.execute('INSERT OR REPLACE INTO document_data VALUES (?, ?, ?, ?, ?)',
                             (str(uuid.uuid4()), phone_number, name, address, details))
                inserted_doc_rows += 1
                print(f"Document row {index} inserted: {phone_number}, {name}")
                
                spam_status = 'Not Spam'
                spam_score = 0
                spam_reports = 0
                c_phone.execute('INSERT OR REPLACE INTO phone_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                               (phone_number, name, address, spam_status, datetime.now().strftime('%H:%M'), details, spam_score, spam_reports))
                inserted_phone_rows += 1
                print(f"Phone row {index} inserted: {phone_number}, {name}")
                
            except Exception as e:
                print(f"Row {index} insert failed: {row.to_dict()}, Error: {str(e)}")
        
        c_doc.execute('COMMIT')
        c_phone.execute('COMMIT')
        conn_doc.close()
        conn_phone.close()
        return {'status': 'Success', 'message': f'{inserted_doc_rows} rows inserted into document data, {inserted_phone_rows} rows into phone data'}
    except Exception as e:
        return {'status': 'Error', 'message': f'Error processing file: {str(e)}'}

# Search document data
def search_document_data(search_term, search_type):
    conn = sqlite3.connect('document_data.db')
    c = conn.cursor()
    search_term = search_term.strip()
    if search_type == 'phone_number':
        c.execute('SELECT * FROM document_data WHERE phone_number = ?', (search_term,))
    elif search_type == 'name':
        c.execute('SELECT * FROM document_data WHERE UPPER(name) LIKE UPPER(?)', (f'%{search_term}%',))
    elif search_type == 'address':
        c.execute('SELECT * FROM document_data WHERE UPPER(address) LIKE UPPER(?)', (f'%{search_term}%',))
    results = c.fetchall()
    conn.close()
    print(f"Search term: {search_term}, Type: {search_type}, Results: {results}")
    return [{'id': r[0], 'phone_number': r[1], 'name': r[2], 'address': r[3], 'details': r[4]} for r in results]

# Check hack status
def check_hack_status(phone_number):
    conn = sqlite3.connect('security_logs.db')
    c = conn.cursor()
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')
    c.execute('SELECT event_time, event_type, description FROM security_logs WHERE phone_number = ? AND event_time >= ?',
             (phone_number, three_days_ago))
    logs = c.fetchall()
    conn.close()
    
    suspicious_events = [log for log in logs if 'Unauthorized' in log[2] or 'Suspicious' in log[2]]
    if suspicious_events:
        return {
            'status': 'Possibly Hacked',
            'details': [{'time': log[0], 'type': log[1], 'description': log[2]} for log in suspicious_events]
        }
    return {'status': 'Not Hacked', 'details': []}

# Get call history
def get_call_history(phone_number):
    conn = sqlite3.connect('call_history.db')
    c = conn.cursor()
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')
    c.execute('SELECT call_time, call_type, contact_number, duration FROM call_history WHERE phone_number = ? AND call_time >= ?',
             (phone_number, three_days_ago))
    calls = c.fetchall()
    conn.close()
    return [{'time': call[0], 'type': call[1], 'contact': call[2], 'duration': call[3]} for call in calls]

# BERT Model
MODEL_PATH = 'trained_bert_model'
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
except:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Binary: spam/ham

def train_bert_model():
    dataset_file = download_sms_spam_dataset()
    df = pd.read_csv(dataset_file, sep='\t', names=['label', 'text'], encoding='utf-8')
    df['label'] = df['label'].map({'ham': 1, 'spam': 0})  # ham=personal, spam=spam
    
    inputs = tokenizer(df['text'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
    labels = torch.tensor(df['label'].tolist())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    batch_size = 16
    for epoch in range(3):  # 3 epochs for better training
        for i in range(0, len(df), batch_size):
            batch_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
            batch_labels = labels[i:i+batch_size]
            outputs = model(**batch_inputs, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss.item()}")
    model.eval()
    
    # Save the trained model
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

# Train model only if not already trained
if not os.path.exists(MODEL_PATH):
    train_bert_model()

def analyze_message_with_bert(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label_idx = torch.argmax(probs, dim=1).item()
    labels = ['Spam', 'Personal']
    confidence = probs[0][label_idx].item()
    
    spam_keywords = ['win', 'free', 'claim', 'urgent', 'prize', 'offer', 'click', 'verify', 'account', 'login', 'password']
    safety_status = 'Safe'
    if any(keyword in text.lower() for keyword in spam_keywords) or label_idx == 0:
        safety_status = 'Not Safe'
    
    return labels[label_idx], safety_status, confidence

# Initialize databases
init_db()
populate_dummy_data()
populate_security_logs()
populate_call_history()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_phone', methods=['POST'])
def analyze_phone():
    phone_number = request.form['phone_number']
    
    conn = sqlite3.connect('phone_data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM phone_data WHERE phone_number = ?', (phone_number,))
    result = c.fetchone()
    conn.close()
    
    if result:
        owner_name, location, spam_status, active_time, message_type, spam_score, spam_reports = result[1:]
    else:
        owner_name = 'Unknown'
        location = 'Unknown'
        spam_status = 'Unknown'
        active_time = datetime.now().strftime('%H:%M')
        message_type = 'Unknown'
        spam_score = 0
        spam_reports = 0
    
    hack_status = check_hack_status(phone_number)
    call_history = get_call_history(phone_number)
    
    return render_template('phone_results.html',
                          phone_number=phone_number,
                          owner_name=owner_name,
                          location=location,
                          spam_status=spam_status,
                          active_time=active_time,
                          message_type=message_type,
                          spam_score=spam_score,
                          spam_reports=spam_reports,
                          hack_status=hack_status,
                          call_history=call_history)

@app.route('/analyze_message', methods=['POST'])
def analyze_message():
    sms_text = request.form['sms_text']
    
    message_type, safety_status, confidence = analyze_message_with_bert(sms_text)
    
    message_details = {
        'sender': 'Unknown',
        'sent_time': datetime.now().strftime('%H:%M')
    }
    
    return render_template('message_results.html',
                          sms_text=sms_text,
                          message_type=message_type,
                          safety_status=safety_status,
                          sender=message_details['sender'],
                          sent_time=message_details['sent_time'],
                          confidence=round(confidence * 100, 2))

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'Error', 'message': 'No file provided'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'Error', 'message': 'No file selected'})
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        result = insert_csv_data(file)
        return jsonify(result)
    return jsonify({'status': 'Error', 'message': 'File must be .csv or .xlsx'})

@app.route('/search_document', methods=['POST'])
def search_document():
    search_term = request.form['search_term']
    search_type = request.form['search_type']
    
    results = search_document_data(search_term, search_type)
    
    return render_template('document_results.html',
                          search_term=search_term,
                          search_type=search_type,
                          results=results)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        theme = request.form.get('theme', 'light')
        language = request.form.get('language', 'English')
        return jsonify({'status': 'Settings saved', 'theme': theme, 'language': language})
    return render_template('settings.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_text = request.form.get('feedback')
    return jsonify({'status': 'Feedback submitted', 'feedback': feedback_text})

@app.route('/debug', methods=['GET'])
def debug():
    conn_doc = sqlite3.connect('document_data.db')
    c_doc = conn_doc.cursor()
    c_doc.execute('SELECT * FROM document_data LIMIT 5')
    doc_rows = c_doc.fetchall()
    conn_doc.close()
    
    conn_phone = sqlite3.connect('phone_data.db')
    c_phone = conn_phone.cursor()
    c_phone.execute('SELECT * FROM phone_data LIMIT 5')
    phone_rows = c_phone.fetchall()
    conn_phone.close()
    
    return jsonify({
        'document_data': [{'id': r[0], 'phone_number': r[1], 'name': r[2], 'address': r[3], 'details': r[4]} for r in doc_rows],
        'phone_data': [{'phone_number': r[0], 'owner_name': r[1], 'location': r[2], 'spam_status': r[3]} for r in phone_rows]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)