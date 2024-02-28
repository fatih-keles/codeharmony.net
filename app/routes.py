from flask import render_template
from datetime import datetime
from app import app

@app.route('/')
def index():
    return render_template('index.html', now=datetime.now)

@app.route('/under-the-hood')
def under_the_hood():
    return render_template('under-the-hood.html', now=datetime.now)

@app.route("/health-check", methods=["GET"])
def health_check():
    """Health check"""
    return '{"status":"OK"}'

if __name__ == '__main__':
    app.run(debug=True)