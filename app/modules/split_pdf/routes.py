from flask import (
    render_template, request, after_this_request, send_file, redirect, url_for, flash
)
import tempfile
import fitz  # PyMuPDF for rendering first page as PDF
import os
from base64 import b64encode
import json
from io import BytesIO
import zipfile
import random
import string

from . import split_pdf_bp

MAX_PAGE_THUMBNAIL_COUNT = 9

@split_pdf_bp.route('/split-pdf')
def index():
    return render_template('split_pdf/index.html')

@split_pdf_bp.route('/split-pdf/get-thumbnails', methods=['POST'])
def get_thumbnails():
    """Get thumbnail image of uploaded PDF file"""
    # get uploaded file
    file = request.files["file"]
    json_response = {
        "status": "success",
        "description": "Successfuly created thumbnail image",
        "pages": []
    }
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf_file:
        try:
            # Write data to the temporary file
            temp_pdf_file.write(file.read())
            temp_pdf_file.flush()
            
            i = 0;
            # Open PDF file 
            pdf_document = fitz.open(temp_pdf_file)
            # Iterate through the pages of the PDF document
            for page in pdf_document.pages():
                page_number = page.number
                page_content = "" #page.get_text()

                # Print the page number and content
                print(f"Page {page_number}: {page_content}")
                # Render the page as an image (PNG format)
                pix = page.get_pixmap()
                # encode in base64
                encoded_image = b64encode(pix.tobytes()).decode("utf-8")
                json_response["pages"].append(
                    {
                        "page_number" : page_number,
                        "page_content" : page_content,
                        "image" : encoded_image
                    }
                )
                # limit to first 10 pages 
                if page_number+1 == MAX_PAGE_THUMBNAIL_COUNT:
                    # with open("./split_pdf/static/document-icon-36553.png", "rb") as image_file:
                    #     # Read the image file and encode it as base64
                    #     encoded_image = b64encode(image_file.read()).decode("utf-8")
                    pages_left = pdf_document.page_count - MAX_PAGE_THUMBNAIL_COUNT
                    json_response["pages"].append(
                        {
                            "page_number" : -1*pdf_document.page_count,
                            "page_content" : f"+{pages_left} Pages",
                            "image" : "encoded_image"
                        }
                    )
                    break;
            
        finally:
            # Delete the temporary file
            os.remove(temp_pdf_file.name)
            print("Temporary files deleted.")
            
    return json.dumps(json_response)


@split_pdf_bp.route('/split-pdf/download-all-pages', methods=['POST'])
def download_all_pages():
    # check if the post request has the file part
    print(f'not in request.filest: {"file" not in request.files}')
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('split_pdf.index'))
    
    # check if file list is empty
    print(f'len(request.files.getlist("file")): {len(request.files.getlist("file"))}')
    if len(request.files.getlist('file')) == 0:
        flash('No files selected')
        return redirect(url_for('split_pdf.index'))
    
    # get uploaded file
    file = request.files["file"]
    print(f'File name: {file}')
    
    # Check upload file size
    pdf_file_stream = BytesIO(file.read())
    file_size = pdf_file_stream.getbuffer().nbytes
    print(f"File Size: {file_size} bytes")
    if file_size == 0:
        flash('No file uploaded, file size 0')
        return redirect(url_for('split_pdf.index'))
    
    # Open the PDF document using fitz.open with the file-like object
    pdf_document = fitz.open(None, pdf_file_stream, "pdf")
    print(f'Page count: {pdf_document.page_count}')
    
    zip_file_name = file.filename + ".zip"
    zip_file_path = os.path.join(tempfile.gettempdir(), get_random_string(16) + "-" +zip_file_name)
        
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        # Iterate through the pages of the PDF document
        for page in pdf_document.pages():
            page_number = page.number
            page_content = "" #page.get_text()
            # create a new PDF
            doc = fitz.open()
            # copy page
            doc.insert_pdf(pdf_document, from_page=page_number, to_page=page_number, start_at=0)
            doc_file_stream = doc.tobytes()
            #tmp_pdf_path = os.path.join(tempfile.gettempdir(), file.filename + '-' + str(page_number) + "_" + ".pdf")
            #doc.save(tmp_pdf_path)
            #doc.close()
            zip_file.writestr(f"{file.filename}-{str(page_number)}.pdf" , doc_file_stream)
    
    @after_this_request
    def remove_file(response):
        try:
            os.remove(zip_file_path)
            print("Temporary files deleted.")
        except Exception as e:
            app.logger.error(f"Error deleting file: {e}")
        return response
    
    #parameter1 = output_file
    return send_file(zip_file_path, as_attachment=True, download_name=zip_file_name)
    #return render_template('split/download.html', parameter1=parameter1)
    
def get_random_string(length):
    """Create random string from all lowercase letter"""
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    #print("Random string of length", length, "is:", result_str)
    return result_str