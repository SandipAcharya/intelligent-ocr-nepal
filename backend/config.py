import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'super_secret_key_idp_nepal')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

    @staticmethod
    def init_app(app):
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
