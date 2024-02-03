from flask import Blueprint

merge_pdf_bp = Blueprint('merge_pdf', __name__, template_folder='templates')

# Import the routes module to make them available
from app.modules.merge_pdf import routes
