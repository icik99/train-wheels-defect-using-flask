from flask import Flask
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'

    # Konfigurasi Supabase
    url: str = os.getenv("SUPABASE_URL")
    key: str = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(url, key)

    app.config['SUPABASE_CLIENT'] = supabase

    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
