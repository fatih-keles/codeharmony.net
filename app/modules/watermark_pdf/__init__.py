from flask import Blueprint

watermark_pdf_bp = Blueprint('watermark_pdf', __name__, template_folder='templates')

# Import the routes module to make them available
from app.modules.watermark_pdf import routes
