import os
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import Config
from pipeline import FaceDetector, DevanagariRecognizer, KYCParser, preprocess_image, perspective_warp
from database import DBManager

app = Flask(__name__)
Config.init_app(app)
CORS(app)

# Initialize ML Models (Load once on startup)
print("Loading ML Models...")
face_detector = FaceDetector()
ocr_recognizer = DevanagariRecognizer()
entity_parser = KYCParser()
db_manager = DBManager()
print("Models loaded successfully.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No document file provided'}), 400
        
    file = request.files['document']
    doc_type = request.form.get('document_type', 'unknown')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # PIPELINE
        try:
            # 1. Preprocess
            orig_img, preprocessed_img = preprocess_image(filepath)
            
            # 2. Layout & Detection
            # Mock warping
            warped_img = perspective_warp(orig_img, corners=None)
            
            # Extract Face
            face_filename = f"face_{filename}"
            face_path = os.path.join(Config.UPLOAD_FOLDER, face_filename)
            extracted_face_path = face_detector.extract_face(orig_img, face_path)
            
            # 3. Recognition
            text_blocks = ocr_recognizer.extract_text(filepath)
            
            # 4. Parsing
            parsed_data = entity_parser.parse(text_blocks)
            parsed_data['document_type'] = doc_type
            
            if extracted_face_path:
                parsed_data['face_image_path'] = extracted_face_path
            
            return jsonify({
                'message': 'Extraction successful',
                'data': parsed_data,
                'raw_text': [block['text'] for block in text_blocks]
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/submit', methods=['POST'])
def submit_form():
    """
    Endpoint for users to submit the verified form after human-in-the-loop correction.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    success = db_manager.insert_document_record(data)
    
    if success:
        return jsonify({'message': 'Data submitted successfully'}), 200
    else:
        return jsonify({'error': 'Failed to save to database'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
