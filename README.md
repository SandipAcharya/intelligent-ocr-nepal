# Intelligent Document Processing (IDP) - Nepal
**Automated KYC & Form Filling Architecture for Complex Devanagari Scripts**

This repository contains the end-to-end implementation of a high-fidelity **Intelligent Document OCR and Automated Form-Filling Engine**. It is specifically architected to process standard Nepali credentials (Nepali Citizenship/Nagarikta, National Identity Card, Driving License) addressing the orthographic and layout complexities of the Devanagari script.

## Problem Domain
Manual form filling across municipal, financial (KYC), and legal sectors in Nepal remains a persistent administrative bottleneck. This manual process is prone to human error, introduces operational delays, and presents accessibility challenges for digitally marginalized citizens. Standard OCR systems (like Tesseract) fail heavily on Devanagari due to the *Shirorekha* (headline connecting characters), multi-tiered vowel modifiers, and cursive conjuncts.

## The Hybrid OCR/HTR Solution Architecture
To resolve these inefficiencies, this system utilizes a highly optimized four-stage deep learning pipeline:

1. **Document Alignment & Rectification (YOLOv8 + OpenCV)**:
   - Uses YOLOv8 (Corner Detection) to locate the quadrilateral boundaries of the document.
   - Calculates a homography matrix for perspective warping, followed by CLAHE contrast normalization and bilateral filtering to denoise the scanned image.
   
2. **Layout Parsing & Portrait Extraction (PP-StructureV3 + MTCNN)**:
   - Parses document layouts into structural blocks.
   - Utilizes a Multi-Task Cascaded Convolutional Network (MTCNN) to automatically detect the owner's portrait, applying a padded bounding box to crop and save the visual asset.

3. **Hybrid Text Recognition Engine (PaddleOCR + TrOCR)**:
   - **Printed Text Labels**: Routed to a Devanagari-optimized **PaddleOCR-VL** model for high-throughput edge inference.
   - **Handwritten User Entries (HTR)**: Variable cursive fields are processed using a transformer-based **TrOCR** (VisionEncoderDecoderModel), which replaces standard LSTM recurrences with cross-attention mechanisms to resolve complex diacritic spatial shifts without requiring character-level segmentation.

4. **Semantic NER Mapping (SpaCy + Regex)**:
   - Passes raw string outputs into a custom SpaCy Named Entity Recognition pipeline to map variables into strict database entities (`FIRST_NAME`, `LAST_NAME`, `CITIZENSHIP_NO`).

5. **Human-in-the-Loop Web Interface (Vanilla JS Glassmorphism)**:
   - Extracted metadata is routed to a dual-pane UI. If CTC/Transformer decoding confidence falls below an acceptable threshold, fields are flagged in red. This validation framework ensures near-perfect submission accuracy to the local MySQL (or SQLite) database.

## Repository Structure
```
intelligent-ocr-nepal/
├── backend/
│   ├── app.py                  # Core Flask REST API (Pydantic / Type Hinting integrated)
│   ├── config.py               # Centralized configuration map
│   ├── database/               # DB manager, pooling, and SQL schemas
│   └── pipeline/               
│       ├── preprocess.py       # OpenCV CLAHE, Bilateral filtering, Homography
│       ├── detector.py         # YOLOv8 layout parser & MTCNN face extractor
│       ├── recognizer.py       # Hybrid Engine: PaddleOCR & HuggingFace TrOCR
│       └── entity_parser.py    # Custom SpaCy NER processing
├── frontend/                   # Vanilla HTML/CSS/JS Glassmorphism Application
├── training/                   # Synthetic Data Generation pipelines (TRDG)
└── docker-compose.yml          # Containerized Orchestration (MySQL + Flask API)
```

## Running the Architecture Locally
The system supports both local python execution and containerized deployments.

### 1. Rapid Local Prototyping (SQLite Fallback)
```bash
# 1. Setup Virtual Environment
python -m venv .venv
source .venv/Scripts/activate  # (Windows: .venv\Scripts\Activate.ps1)

# 2. Install Pipeline Dependencies
pip install -r backend/requirements.txt

# 3. Start the Inference Server
python backend/app.py
```
*Note: PaddleOCR and TrOCR model weights will be automatically downloaded from HuggingFace/Paddle repositories on initial runtime.*

### 2. Frontend Interface
Open `frontend/index.html` directly in any modern browser, or run a local python HTTP server:
```bash
python -m http.server 8080 -d frontend
```
Visit `http://localhost:8080` to access the Human-in-the-Loop validation UI.

### 3. Production Deployment (Docker Compose)
```bash
docker-compose up --build -d
```
This spins up the production-ready MySQL 8.0 instance along with the backend API containers.
