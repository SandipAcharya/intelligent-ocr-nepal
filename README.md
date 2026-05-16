# Intelligent Document Processing (IDP) - Nepal
**Automated KYC & Form Filling for Nepal**

This repository contains the end-to-end implementation of an Intelligent Document OCR and Automated Form-Filling Engine designed for standard Nepali credentials (Nepali Citizenship, National Identity Card, Driving License).

## Problem Statement
Manual form filling is highly prone to human error, introduces operational delays, and presents accessibility challenges for elderly or digitally marginalized citizens in Nepal who frequently experience stress when navigating complex administrative documentation in banking, government licensing, and KYC verifications.

## Solution Architecture
1. **Document Alignment**: Preprocesses and aligns uploaded images using bounding boxes/corner detection and perspective warping.
2. **Layout Parsing & Portrait Extraction**: MTCNN locates the owner's portrait, crops it, and extracts structured layout blocks.
3. **Hybrid OCR/HTR**: Routes printed text labels to EasyOCR/PaddleOCR and handwritten data through specialized recognition pipelines.
4. **Named Entity Recognition (NER)**: Parses text blobs to extract key fields (First Name, Last Name, Citizenship Number, DOB).
5. **Human-in-the-Loop Web UI**: Beautiful, interactive split-pane interface to verify low-confidence fields before saving to the MySQL database.

## Technology Stack
- **Backend**: Flask (Python)
- **Database**: MySQL 8.0
- **Computer Vision**: OpenCV, MTCNN
- **OCR Engine**: EasyOCR (supports Devanagari)
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism design), Vanilla JS
- **Deployment**: Docker & Docker Compose

## Project Structure
```
intelligent-ocr-nepal/
├── backend/
│   ├── app.py                  # API routes
│   ├── config.py               # Env configurations
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── database/               # Database Manager & Schema
│   └── pipeline/               # Core ML Engines (Preprocess, Detector, Recognizer, Parser)
├── frontend/
│   ├── index.html              # Human-in-the-Loop UI
│   ├── css/styles.css
│   └── js/app.js
└── docker-compose.yml          # Container configuration
```

## Setup & Installation

### Local Setup (Without Docker)
1. Navigate to the `backend` directory: `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Flask Server: `python app.py`
4. Open `frontend/index.html` in any modern web browser to use the app.

### Docker Setup
1. Ensure Docker Desktop is running.
2. Run `docker-compose up --build`
3. The backend API will be available at `http://localhost:5000`
4. Open `frontend/index.html` in your browser.

## How It Works
1. Upload an image of a Nepali identity document on the frontend.
2. The image is processed, text is extracted via the backend OCR pipeline.
3. Relevant fields and confidence scores are computed.
4. The user verifies the extracted data in the Glassmorphic UI (low-confidence fields are highlighted).
5. Upon confirmation, data is persisted to the MySQL database.
