from flask import Blueprint

image_tools_bp = Blueprint('image_tools', __name__, template_folder='templates')

# Import the routes module to make them available
from app.modules.image_tools import routes
