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
from PIL import Image, ImageEnhance

from . import watermark_pdf_bp

MAX_PAGE_THUMBNAIL_COUNT = 9

@watermark_pdf_bp.route('/watermark-pdf')
def index():
    return render_template('watermark_pdf/index.html')

@watermark_pdf_bp.route('/watermark-pdf/get-thumbnail', methods=['POST'])
def get_thumbnail():
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


@watermark_pdf_bp.route('/watermark-pdf/download-watermarked-pdf', methods=['POST'])
def download_watermarked_pdf():
    # check if the post request has the file part
    print(f'not in request.filest: {"watermark-pdf-file" not in request.files}')
    if 'watermark-pdf-file' not in request.files:
        flash('Missing PDF file', 'error')
        return redirect(url_for('watermark_pdf.index'))
    print(f'not in request.filest: {"watermark-image-file" not in request.files}')
    if 'watermark-image-file' not in request.files:
        flash('Missing watermark image file', 'error')
        return redirect(url_for('watermark_pdf.index'))
    
    # check if file list is empty
    print(f'len(request.files.getlist("watermark-pdf-file")): {len(request.files.getlist("watermark-pdf-file"))}')
    if len(request.files.getlist('watermark-pdf-file')) == 0:
        flash('No PDF file selected')
        return redirect(url_for('watermark_pdf.index'))
    print(f'len(request.files.getlist("watermark-image-file")): {len(request.files.getlist("watermark-image-file"))}')
    if len(request.files.getlist('watermark-pdf-file')) == 0:
        flash('No watermark image file selected')
        return redirect(url_for('watermark_pdf.index'))
    
    # get uploaded file
    watermark_pdf_file = request.files["watermark-pdf-file"]
    watermark_image_file = request.files["watermark-image-file"]
    print(f'File name: {watermark_pdf_file}')
    print(f'File name: {watermark_image_file}')
    
    # Check upload file size
    pdf_file_stream = BytesIO(watermark_pdf_file.read())
    pdf_file_size = pdf_file_stream.getbuffer().nbytes
    print(f"File Size: {pdf_file_size} bytes")
    if pdf_file_size == 0:
        flash('No PDF file uploaded, file size 0')
        return redirect(url_for('watermark_pdf.index'))
    
    image_file_stream = BytesIO(watermark_image_file.read())
    image_file_size = image_file_stream.getbuffer().nbytes
    print(f"File Size: {image_file_size} bytes")
    if image_file_size == 0:
        flash('No PDF file uploaded, file size 0')
        return redirect(url_for('watermark_pdf.index'))
    
    with tempfile.NamedTemporaryFile(delete=False) as image_temp_file:
        try:
            # Write data to the temporary file
            image_temp_file.write(image_file_stream.getbuffer())
            image_temp_file.flush()
            image_temp_file.close()
            
            image = Image.open(image_temp_file.name)
            # Convert the image to RGBA mode
            image = image.convert('RGBA')
            
            # Create a new image with an alpha channel
            #new_image = Image.new('RGBA', image.size, (255, 255, 255, 255))
            new_image = image
            # Composite the original image onto the new image with transparency
            #new_image.paste(image, (0, 0), image)
            new_image.putalpha(50)
    
            # Rotate the image
            rotated_image = new_image.rotate(45, expand=True)
            # Save or display the rotated image
            rotated_image.save(image_temp_file.name, "PNG")
            
            # Open the PDF document using fitz.open with the file-like object
            pdf_document = fitz.open(None, pdf_file_stream, "pdf")
            print(f'Page count: {pdf_document.page_count}')
            
            pix = fitz.Pixmap(image_temp_file.name)
            # first execution embeds the image
            img_xref = 0 
            # Iterate through the pages of the PDF document
            for page in pdf_document.pages():
                page_number = page.number
                page_content = "" #page.get_text()
                
                # insert an image watermark from a file name to fit the page bounds
                # page.insert_image(page.bound(),filename="watermark.png", overlay=False)
                # page.insert_image(page.bound(), stream=image_file_stream, xref=img_xref, overlay=False)
                r = page.bound()
                #print(f"{r}")
                m = 20
                page_bound = fitz.Rect(r.x0+m, r.y0+m, r.x1-m, r.y1-m)
                page.insert_image(page_bound, pixmap=pix, xref=img_xref, overlay=True)
        finally:
            # Delete the temporary file
            os.remove(image_temp_file.name)
            print("Temporary files deleted.")
        
    #pdf_document.save("/tmp/test.pdf", clean=True, deflate=True, deflate_images=True, deflate_fonts=True)
    
    return send_file(BytesIO(pdf_document.tobytes()), as_attachment=True, download_name=watermark_pdf_file.filename)
    #return render_template('split/download.html', parameter1=parameter1)
    
def get_random_string(length):
    """Create random string from all lowercase letter"""
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    #print("Random string of length", length, "is:", result_str)
    return result_str

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