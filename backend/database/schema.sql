CREATE TABLE IF NOT EXISTS extracted_documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_type VARCHAR(50) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    citizenship_no VARCHAR(50),
    date_of_birth VARCHAR(50),
    address VARCHAR(255),
    face_image_path VARCHAR(255),
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
