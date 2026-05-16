import sqlite3
import os

class DBManager:
    def __init__(self):
        # We use a local SQLite database for easy testing without Docker
        self.db_path = os.path.join(os.path.dirname(__file__), 'idp_nepal.db')
        self.init_db()
        
    def get_connection(self):
        try:
            connection = sqlite3.connect(self.db_path)
            return connection
        except Exception as e:
            print(f"Error connecting to SQLite: {e}")
            return None

    def init_db(self):
        conn = self.get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS extracted_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_type TEXT NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    citizenship_no TEXT,
                    date_of_birth TEXT,
                    address TEXT,
                    face_image_path TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()

    def insert_document_record(self, data):
        conn = self.get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        query = """
            INSERT INTO extracted_documents 
            (document_type, first_name, last_name, citizenship_no, date_of_birth, address, face_image_path, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            data.get('document_type', 'unknown'),
            data.get('first_name', ''),
            data.get('last_name', ''),
            data.get('citizenship_no', ''),
            data.get('date_of_birth', ''),
            data.get('address', ''),
            data.get('face_image_path', ''),
            data.get('confidence_score', 0.0)
        )
        try:
            cursor.execute(query, values)
            conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting record: {e}")
            return False
        finally:
            if conn:
                conn.close()
