import os
import cv2
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import Config
from pipeline import DocumentDetector, HybridRecognizer, KYCParser, preprocess_image, perspective_warp
from database import DBManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
Config.init_app(app)
CORS(app)

logger.info("Initializing ML Models...")
try:
    doc_detector = DocumentDetector()
    ocr_engine = HybridRecognizer(use_gpu=False)  # Set True if CUDA is available
    entity_parser = KYCParser()
    db_manager = DBManager()
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize models: {e}")

def allowed_file(filename: str) -> bool:
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
        
        try:
            logger.info(f"Processing {doc_type} document: {filename}")
            
            # 1. Preprocessing (Grayscale, CLAHE, Bilateral Filter)
            orig_img, preprocessed_img = preprocess_image(filepath)
            
            # 2. Quadrilateral Detection & Perspective Warping (YOLOv8 & Homography)
            corners = doc_detector.detect_corners(orig_img)
            warped_img = perspective_warp(orig_img, corners=corners)
            
            # 3. Portrait Extraction (MTCNN)
            face_filename = f"face_{filename}"
            face_path = os.path.join(Config.UPLOAD_FOLDER, face_filename)
            extracted_face_path = doc_detector.extract_portrait(orig_img, face_path)
            
            # 4. Hybrid OCR/HTR Extraction (PaddleOCR & TrOCR)
            text_blocks = ocr_engine.process_document(filepath)
            
            # 5. Named Entity Recognition
            parsed_data = entity_parser.parse(text_blocks)
            parsed_data['document_type'] = doc_type
            
            if extracted_face_path:
                parsed_data['face_image_path'] = extracted_face_path
            
            logger.info("Document processing completed successfully.")
            return jsonify({
                'message': 'Extraction successful',
                'data': parsed_data,
                'raw_text': [block['text'] for block in text_blocks]
            }), 200
            
        except Exception as e:
            logger.error(f"Pipeline failure: {e}")
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
        logger.info("Record verified and saved to database.")
        return jsonify({'message': 'Data submitted successfully'}), 200
    else:
        logger.error("Database insertion failed.")
        return jsonify({'error': 'Failed to save to database'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)
