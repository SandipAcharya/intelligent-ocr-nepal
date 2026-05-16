import mysql.connector
from mysql.connector import Error
import os

class DBManager:
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.user = os.getenv('DB_USER', 'idp_user')
        self.password = os.getenv('DB_PASSWORD', 'idp_password')
        self.database = os.getenv('DB_NAME', 'idp_nepal')
        
    def get_connection(self):
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None

    def insert_document_record(self, data):
        conn = self.get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        query = """
            INSERT INTO extracted_documents 
            (document_type, first_name, last_name, citizenship_no, date_of_birth, address, face_image_path, confidence_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
        except Error as e:
            print(f"Error inserting record: {e}")
            return False
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
