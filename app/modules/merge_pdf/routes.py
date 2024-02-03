from flask import (
    render_template, request, after_this_request, send_file, redirect, url_for, flash
)
from werkzeug.utils import secure_filename
import tempfile
import fitz  # PyMuPDF for rendering first page as PDF
import os
from base64 import b64encode
import json
from io import BytesIO
import zipfile
import random
import string

from . import merge_pdf_bp

MAX_PAGE_THUMBNAIL_COUNT = 9

@merge_pdf_bp.route('/merge-pdf')
def index():
    return render_template('merge_pdf/index.html')

@merge_pdf_bp.route('/merge-pdf/get-thumbnails', methods=['POST'])
def get_thumbnails():
    """Get thumbnail image of uploaded PDF file"""
    # get uploaded file
    file = request.files["file"]
    json_response = {
        "status": "success",
        "description": "Successfuly created thumbnail image",
        "image" : ""
    }
    
    # Temporary PNG file name
    tmp_png_path = os.path.join(tempfile.gettempdir(), 'get-thumbnail-' + get_random_string(16)+".png")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf_file:
        try:
            # Write data to the temporary file
            temp_pdf_file.write(file.read())
            temp_pdf_file.flush()

            # Perform some processing on the temporary file 
            pdf_to_image(temp_pdf_file, tmp_png_path)
            json_response["image"] = file_to_base64(tmp_png_path)
            
            # Display the processed data
            # print("Processed Data:", processed_data)
        finally:
            # Delete the temporary file
            os.remove(temp_pdf_file.name)
            os.remove(tmp_png_path)
            print("Temporary files deleted.")
            
    return json.dumps(json_response)

@merge_pdf_bp.route('/merge-pdf/download-merged-file', methods=['POST'])
def download_merged_file():
    # check if the post request has the file part
    print(f'not in request.filest: {"files" not in request.files}')
    if 'files' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('merge_pdf.index'))
    
    # check if file list is empty
    print(f'len(request.files.getlist("files")): {len(request.files.getlist("files"))}')
    if len(request.files.getlist('files')) == 0:
        flash('No files selected')
        return redirect(url_for('merge_pdf.index'))

    # Retrieve uploaded files
    files = request.files.getlist('files')
    print(f'File List: {files}')
    
    tmp_pdf_files = []
    # Validate uploaded files
    for file in files:
        print(f'File name: {file.filename}')
        if secure_filename(file.filename) != '' and file.content_type in ['application/pdf']:
            #new_filename = secure_filename(file.filename)
            tmp_pdf_path = os.path.join(tempfile.gettempdir(), 'merge_pdf-' + get_random_string(16)+".pdf")
            print(f'Secure File name: {tmp_pdf_path}')
            # Save the file to a temporary or permanent storage location
            file.save(os.path.join(tempfile.gettempdir(), tmp_pdf_path))
            tmp_pdf_files.append(tmp_pdf_path)

    # Check if uploaded file list is empty
    if len(tmp_pdf_files) == 0:
        flash('No suitable files uploaded')
        return redirect(url_for('merge_pdf.index'))

    # Create a PDF writer object
    pdf_writer = fitz.open()
    # Iterate over the PDF paths and add them to the writer
    for pdf_path in tmp_pdf_files:
        pdf_reader = fitz.open(pdf_path)
        pdf_writer.insert_pdf(pdf_reader)

    # Save the merged PDF to the output path
    output_file = 'merge_pdf-' + get_random_string(16)+".pdf"
    output_path = os.path.join(tempfile.gettempdir(), output_file)
    pdf_writer.save(output_path, garbage=4, deflate=True)
    pdf_writer.close()
    tmp_pdf_files.append(output_path)
    
    @after_this_request
    def remove_file(response):
        try:
            os.remove(output_path)
            print("Temporary files deleted.")
        except Exception as e:
            app.logger.error(f"Error deleting file: {e}")
        return response
    
    return send_file(output_path, as_attachment=True, download_name="merged-pdf.pdf")
    # parameter1 = output_file
    # return render_template('merge/download.html', parameter1=parameter1)

def file_to_base64(file_path):
    """Get Base64 encoded file contents"""
    try:
        with open(file_path, "rb") as image_file:
            # Read the image file and encode it as base64
            base64_encoded = b64encode(image_file.read()).decode("utf-8")
            return base64_encoded
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None

def pdf_to_image(pdf_path, image_path):
    """ Get first page of PDF file and create an image """
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    # Get the first page
    first_page = pdf_document[0]
    # Render the page as an image (PNG format)
    pix = first_page.get_pixmap()
    # Save the image
    pix.save(image_path)

def get_random_string(length):
    """Create random string from all lowercase letter"""
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    #print("Random string of length", length, "is:", result_str)
    return result_str