from flask import Blueprint

example_bp = Blueprint('example', __name__, template_folder='templates')

# Import the routes module to make them available
from app.modules.example import routes
