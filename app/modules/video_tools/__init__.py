from flask import Blueprint

video_tools_bp = Blueprint('video_tools', __name__, template_folder='templates')

# Import the routes module to make them available
from app.modules.video_tools import routes

# stream_log = None
# stream_output = None