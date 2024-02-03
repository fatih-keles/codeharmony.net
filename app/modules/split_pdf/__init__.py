from flask import Blueprint

split_pdf_bp = Blueprint('split_pdf', __name__, template_folder='templates')

# Import the routes module to make them available
from app.modules.split_pdf import routes
