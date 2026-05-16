import os
import cv2
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import Config
from pipeline import DocumentDetector, HybridRecognizer, KYCParser, preprocess_image, perspective_warp
from database import DBManager

# ─────────────────────────── Logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────── Flask App ───────────────────────────
# Serve frontend/index.html directly from Flask so there's only ONE server to run
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
Config.init_app(app)
CORS(app)

# ─────────────────────────── Model Init ──────────────────────────
logger.info("Initializing ML Models...")
try:
    doc_detector = DocumentDetector()
    ocr_engine = HybridRecognizer(use_gpu=False)
    entity_parser = KYCParser()
    db_manager = DBManager()
    logger.info("All models loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize models: {e}")
    doc_detector = None
    ocr_engine = None
    entity_parser = KYCParser()
    db_manager = DBManager()

# ─────────────────────────── Routes ──────────────────────────────

@app.route('/')
def serve_frontend():
    """Serve the Human-in-the-Loop web interface."""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve frontend static assets (CSS, JS)."""
    return send_from_directory(FRONTEND_DIR, path)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """
    Pipeline Endpoint:
      1. Preprocess (CLAHE + Bilateral Filter)
      2. Corner Detection (YOLOv8) + Homography Warp
      3. Portrait Extraction (MTCNN)
      4. Hybrid OCR: PaddleOCR (printed) + TrOCR (handwritten)
      5. NER Parsing (SpaCy + RegEx)
    """
    if 'document' not in request.files:
        return jsonify({'error': 'No document file provided'}), 400

    file = request.files['document']
    doc_type = request.form.get('document_type', 'unknown')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file format. Use JPG, PNG, or PDF.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        logger.info(f"Processing [{doc_type}]: {filename}")

        # Stage 1: Preprocessing
        orig_img, _ = preprocess_image(filepath)

        # Stage 2: Layout Detection & Portrait Extraction
        face_path = None
        if doc_detector:
            corners = doc_detector.detect_corners(orig_img)
            perspective_warp(orig_img, corners=corners)

            face_filename = f"face_{filename}"
            face_filepath = os.path.join(Config.UPLOAD_FOLDER, face_filename)
            face_path = doc_detector.extract_portrait(orig_img, face_filepath)

        # Stage 3: Hybrid OCR/HTR
        text_blocks = []
        if ocr_engine:
            text_blocks = ocr_engine.process_document(filepath)
        else:
            logger.warning("OCR engine unavailable – returning empty extraction.")

        # Stage 4: Named Entity Recognition / Dynamic Extraction
        parsed_data = entity_parser.parse(text_blocks, doc_type=doc_type)
        parsed_data['document_type'] = doc_type
        if face_path:
            parsed_data['face_image_path'] = face_path

        logger.info(f"Extraction complete. Confidence: {parsed_data.get('confidence_score', 0):.2f}")
        return jsonify({
            'message': 'Extraction successful',
            'data': parsed_data,
            'raw_text': [block['text'] for block in text_blocks]
        }), 200

    except Exception as e:
        logger.error(f"Pipeline failure on {filename}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/submit', methods=['POST'])
def submit_form():
    """
    Human-in-the-Loop submission: saves user-verified data to the database.
    All manual corrections are logged for continuous model improvement.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    success = db_manager.insert_document_record(data)
    if success:
        logger.info(f"Record submitted: {data.get('citizenship_no', '?')}")
        return jsonify({'message': 'Data verified and submitted successfully'}), 200
    else:
        return jsonify({'error': 'Database insertion failed'}), 500


if __name__ == '__main__':
    logger.info(f"Serving frontend from: {FRONTEND_DIR}")
    logger.info("Starting Flask on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
