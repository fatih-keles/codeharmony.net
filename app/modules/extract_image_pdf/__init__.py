from flask import Blueprint

extract_image_pdf_bp = Blueprint('extract_image_pdf', __name__, template_folder='templates')

# Import the routes module to make them available
from app.modules.extract_image_pdf import routes
