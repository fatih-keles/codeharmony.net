from flask import Flask
import os
import secrets

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))

# Import your routes after creating the Flask app instance
from app import routes

# Import and register the example blueprint
from app.modules.example.routes import example_bp
app.register_blueprint(example_bp)

# Import and register the split_pdf blueprint
from app.modules.split_pdf.routes import split_pdf_bp
app.register_blueprint(split_pdf_bp)

# Import and register the merge_pdf blueprint
from app.modules.merge_pdf.routes import merge_pdf_bp
app.register_blueprint(merge_pdf_bp)

# Import and register the extract_image_pdf blueprint
from app.modules.extract_image_pdf.routes import extract_image_pdf_bp
app.register_blueprint(extract_image_pdf_bp)

# Import and register the watermark_pdf blueprint
from app.modules.watermark_pdf.routes import watermark_pdf_bp
app.register_blueprint(watermark_pdf_bp)

# Import and register the image_tools blueprint
from app.modules.image_tools.routes import image_tools_bp
app.register_blueprint(image_tools_bp)