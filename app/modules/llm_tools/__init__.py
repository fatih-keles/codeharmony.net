from flask import Blueprint

llm_tools_bp = Blueprint('llm_tools', __name__, template_folder='templates')

# Import the routes module to make them available
from app.modules.llm_tools import routes

stream_log = None
stream_output = None