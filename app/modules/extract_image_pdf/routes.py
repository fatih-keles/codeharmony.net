from flask import (
    render_template, request, after_this_request, send_file, redirect, url_for, flash
)
import tempfile
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF for rendering first page as PDF
import os
from base64 import b64encode
import json
from io import BytesIO
import zipfile
import random
import string

from . import extract_image_pdf_bp

MAX_PAGE_THUMBNAIL_COUNT = 8

@extract_image_pdf_bp.route('/extract-image-pdf')
def index():
    return render_template('extract_image_pdf/index.html')

@extract_image_pdf_bp.route('/extract-image-pdf/get-thumbnails', methods=['POST'])
def get_thumbnails():
    """Get thumbnail image of uploaded PDF file"""
    # get uploaded file
    file = request.files["file"]
    json_response = {
        "status": "success",
        "description": "Successfuly created thumbnail images",
        "images": []
    }
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf_file:
        try:
            # Write data to the temporary file
            temp_pdf_file.write(file.read())
            temp_pdf_file.flush()
            
            i = 0;
            break_flag = False
            # Open PDF file 
            pdf_document = fitz.open(temp_pdf_file)
            # Iterate through the pages of the PDF document
            for page_index in range(len(pdf_document)):
                page = pdf_document[page_index] # get the page
                page_number = page.number
                page_content = "" #page.get_text()

                # Print the page number and content
                print(f"Page {page_number}: {page_content}")
                image_list = page.get_images()

                # print the number of images found on the page
                if image_list:
                    print(f"Found {len(image_list)} images on page {page_index}")
                else:
                    print("No images found on page", page_index)
                    
                for image_index, img in enumerate(image_list, start=1): # enumerate the image list
                    xref = img[0] # get the XREF of the image
                    pix = fitz.Pixmap(pdf_document, xref) # create a Pixmap

                    if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    # encode in base64
                    encoded_image = b64encode(pix.tobytes()).decode("utf-8")
                    json_response["images"].append(
                        {
                            "page_number" : page_number,
                            "page_content" : page_content,
                            "image_index": image_index,
                            "image" : encoded_image
                        }
                    )
                    # pix.save("page_%s-image_%s.png" % (page_index, image_index)) # save the image as png
                    pix = None
                    
                    # limit to first 10 pages 
                    if len(json_response["images"])+1 == MAX_PAGE_THUMBNAIL_COUNT:
                        # with open("./split_pdf/static/document-icon-36553.png", "rb") as image_file:
                        #     # Read the image file and encode it as base64
                        #     encoded_image = b64encode(image_file.read()).decode("utf-8")
                        pages_left = len(pdf_document) - MAX_PAGE_THUMBNAIL_COUNT
                        json_response["images"].append(
                            {
                                "page_number" : -1*len(pdf_document),
                                "page_content" : f"+More ",
                                "image_index" : -1*len(pdf_document),
                                "image" : ""
                            }
                        )
                        break_flag = True
                        break
                    
                if break_flag:
                    break
                    
        finally:
            # Delete the temporary file
            os.remove(temp_pdf_file.name)
            print("Temporary files deleted.")
            
    return json.dumps(json_response)


@extract_image_pdf_bp.route('/extract-image-pdf/download-all-images', methods=['POST'])
def download_all_images():
    # check if the post request has the file part
    print(f'not in request.filest: {"file" not in request.files}')
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('extract_image_pdf.index'))
    
    # check if file list is empty
    print(f'len(request.files.getlist("file")): {len(request.files.getlist("file"))}')
    if len(request.files.getlist('file')) == 0:
        flash('No files selected')
        return redirect(url_for('extract_image_pdf.index'))
    
    # get uploaded file
    file = request.files["file"]
    print(f'File name: {file}')
    
    # Check upload file size
    pdf_file_stream = BytesIO(file.read())
    file_size = pdf_file_stream.getbuffer().nbytes
    print(f"File Size: {file_size} bytes")
    if file_size == 0:
        flash('No file uploaded, file size 0')
        return redirect(url_for('extract_image_pdf.index'))
    
    # Open the PDF document using fitz.open with the file-like object
    pdf_document = fitz.open(None, pdf_file_stream, "pdf")
    print(f'Page count: {pdf_document.page_count}')
    
    zip_file_name = secure_filename(file.filename) + ".zip"
    zip_file_path = os.path.join(tempfile.gettempdir(), get_random_string(16) + "-" +zip_file_name)
        
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        image_count = 0
        # Iterate through the pages of the PDF document
        for page in pdf_document.pages():
            page_number = page.number
            page_content = "" #page.get_text()
            # Images on the page
            image_list = page.get_images()
            
            for image_index, img in enumerate(image_list, start=1): # enumerate the image list
                xref = img[0] # get the XREF of the image
                pix = fitz.Pixmap(pdf_document, xref) # create a Pixmap

                if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                image_stream = pix.tobytes()
                zip_file.writestr(f"{image_count}.png" , image_stream)
                image_count = image_count + 1
    
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