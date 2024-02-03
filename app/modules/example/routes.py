from flask import render_template
from . import example_bp

@example_bp.route('/example')
def index():
    return render_template('example/index.html')
