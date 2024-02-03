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
from PIL import Image, ImageDraw, ExifTags
import datetime
import time
from app.modules.image_tools.mandelbrot import MandelbrotSet
from app.modules.image_tools.viewport import Viewport, Preset
import matplotlib.cm
from app.modules.image_tools.helper import Helper
import numpy as np
from random import shuffle
import random
from PIL import Image
import matplotlib.pyplot as plt
import squarify 
import seaborn as sns

from . import image_tools_bp

IMAGE_BLACK_AND_WHITE = "1"
IMAGE_GRAYSCALE = "L"
IMAGE_COLOR = "RGB"
CODEHARMONY = True
MAX_PUNCH_DEPTH = 9
MAX_MANDELBROT_ITERATIONS = 2000
DEFAULT_COLOR_PALETTE_SIZE = 16
MAX_COLOR_PALETTE_SIZE = 256

@image_tools_bp.route('/fractal')
def fractal():
    return render_template('image_tools/fractal.html', default_presets=Viewport.default_presets)

@image_tools_bp.route('/fractal/get-punch', methods=['POST'])
def get_punch():
    """Get thumbnail image of uploaded PDF file"""
    # get uploaded file
    depth = request.form.get("depth")
    bgcolor = request.form.get("bgcolor")
    fgcolor = request.form.get("fgcolor")
    usecompcolor = True if request.form.get("usecompcolor") == "true" else False
    userandomcolor = True if request.form.get("userandomcolor") == "true" else False
    json_response = {
        "status": "success",
        "description": "Successfuly created fractal image",
        "image" : ""
    }
    # safe guard depth
    n_depth = MAX_PUNCH_DEPTH
    try:
        n_depth = int(depth)
        if n_depth > MAX_PUNCH_DEPTH:
            n_depth = MAX_PUNCH_DEPTH
    except:
        n_depth = MAX_PUNCH_DEPTH
    
    punch_color = None
    if usecompcolor:
        punch_color = tuple(255 - c for c in hex_to_tuple(bgcolor))
    else:
        punch_color = hex_to_tuple(fgcolor)
        
    pix = generate_punch2(depth=n_depth, bgcolor=hex_to_tuple(bgcolor), fgcolor=punch_color, use_random_colors=userandomcolor)
    
    encoded_image = None
    if CODEHARMONY:
        image = Image.open(BytesIO(pix.tobytes()))
        image = codeharmony(image)
        image_buffer = BytesIO()
        image.save(image_buffer, "PNG")
        encoded_image = b64encode(image_buffer.getbuffer()).decode("utf-8")
    else:
        encoded_image = b64encode(pix.tobytes()).decode("utf-8")
        
    # pix = generate_mandelbrot(600, 600, 500)
    # encoded_image = b64encode(pix.tobytes()).decode("utf-8")
    json_response["image"] = encoded_image
    return json.dumps(json_response)

def hex_to_tuple(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

def codeharmony(image):
    height = image.height
    width = image.width
    
    text = "codeharmony.net"
    text_position = (2, height-12)
    
    d = ImageDraw.Draw(image)
    d.text(text=text, xy=text_position, fill="red")
    
    exif = image.getexif() 
    exif[ExifTags.Base.Software] = 'CodeHarmony using Pillow'
    exif[ExifTags.Base.DateTime] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")

    return image

def generate_punch(depth, bgcolor, fgcolor):
    n = depth                         # depth (precision)
    d = 3**n                          # edge length

    t0 = time.perf_counter()
    ir = (0, 0, d, d)                 # the pixmap rectangle

    pm = fitz.Pixmap(fitz.csRGB, ir, False)
    # pm.set_rect(pm.irect, (255,255,0)) # fill it with some background color
    pm.set_rect(pm.irect, bgcolor) 

    # color = (0, 0, 255)               # color to fill the punch holes
    color = fgcolor

    # alternatively, define a 'fill' pixmap for the punch holes
    # this could be anything, e.g. some photo image ...
    fill = fitz.Pixmap(fitz.csRGB, ir, False) # same size as 'pm'
    fill.set_rect(fill.irect, (0, 255, 255))   # put some color in

    def punch(x, y, step):
        """Recursively "punch a hole" in the central square of a pixmap.
        Arguments are top-left coords and the step width.
        Some alternative punching methods are commented out.
        """
        s = step // 3                 # the new step
        # iterate through the 9 sub-squares
        # the central one will be filled with the color
        for i in range(3):
            for j in range(3):
                if i != j or i != 1:  # this is not the central cube
                    if s >= 3:        # recursing needed?
                        punch(x+i*s, y+j*s, s)       # recurse
                else:                 # punching alternatives are:
                    pm.set_rect((x+s, y+s, x+2*s, y+2*s), color)     # fill with a color
                    # pm.copy(fill, (x+s, y+s, x+2*s, y+2*s))  # copy from fill
                    # pm.invert_irect((x+s, y+s, x+2*s, y+2*s))       # invert colors
        return

    #==============================================================================
    # main program
    #==============================================================================
    # now start punching holes into the pixmap
    punch(0, 0, d)
    t1 = time.perf_counter()
    # pm.save("/tmp/sierpinski-punch"+str(n)+".png")
    # t2 = time.perf_counter()
    print ("%g sec to create / fill the pixmap" % round(t1-t0,3))
    # print ("%g sec to save the image" % round(t2-t1,3))
    return pm

def generate_punch2(depth, bgcolor, fgcolor, use_random_colors=False):
    n = depth                         # depth (precision)
    d = 3**n                          # edge length

    t0 = time.perf_counter()
    ir = (0, 0, d, d)                 # the pixmap rectangle

    pm = fitz.Pixmap(fitz.csRGB, ir, False)
    # pm.set_rect(pm.irect, (255,255,0)) # fill it with some background color
    pm.set_rect(pm.irect, bgcolor) 
    
    # color = (0, 0, 255)               # color to fill the punch holes
    color = fgcolor
    
    # alternatively, define a 'fill' pixmap for the punch holes
    # this could be anything, e.g. some photo image ...
    fill = fitz.Pixmap(fitz.csRGB, ir, False) # same size as 'pm'
    fill.set_rect(fill.irect, (0, 255, 255))   # put some color in

    def punch(x, y, step):
        """Recursively "punch a hole" in the central square of a pixmap.
        Arguments are top-left coords and the step width.
        Some alternative punching methods are commented out.
        """
        s = step // 3                 # the new step
        # iterate through the 9 sub-squares
        # the central one will be filled with the color
        for i in range(3):
            for j in range(3):
                if i != j or i != 1:  # this is not the central cube
                    if s >= 3:        # recursing needed?
                        punch(x+i*s, y+j*s, s)       # recurse
                else:                 # punching alternatives are:
                    if use_random_colors:
                        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        pm.set_rect((x+s, y+s, x+2*s, y+2*s), random_color)     # change color
                    else:
                        pm.set_rect((x+s, y+s, x+2*s, y+2*s), color)     # fill with a color
                    # pm.copy(fill, (x+s, y+s, x+2*s, y+2*s))  # copy from fill
                    # pm.invert_irect((x+s, y+s, x+2*s, y+2*s))       # invert colors
        return

    #==============================================================================
    # main program
    #==============================================================================
    # now start punching holes into the pixmap
    punch(0, 0, d)
    t1 = time.perf_counter()
    # pm.save("/tmp/sierpinski-punch"+str(n)+".png")
    # t2 = time.perf_counter()
    print ("%g sec to create / fill the pixmap" % round(t1-t0,3))
    # print ("%g sec to save the image" % round(t2-t1,3))
    return pm

@image_tools_bp.route('/fractal/get-mandelbrot', methods=['POST'])
def get_mandelbrot():
    """Get thumbnail image of uploaded PDF file"""
    json_response = {
        "status": "success",
        "description": "Successfuly created fractal image",
        "image" : ""
    }
    # get image type
    f_imgtype = request.form.get("imgtype")
    print(f"f_imgtype : {f_imgtype}")
    imgtype = IMAGE_BLACK_AND_WHITE
    if f_imgtype == "blackwhite":
        imgtype = IMAGE_BLACK_AND_WHITE
    if f_imgtype == "grayscale":
        imgtype = IMAGE_GRAYSCALE
    if f_imgtype == "color":
        imgtype = IMAGE_COLOR
    
    # get color map name
    colormap_name = request.form.get("colormap_name")
    cmexterior = request.form.get("cmexterior")
    cmexterior_smooth = request.form.get("cmexterior_smooth")
    
    cminterior = request.form.get("cminterior")
    cminterior_smooth = request.form.get("cminterior_smooth")
    
    cmgray = request.form.get("cmgray")
    cmgray_smooth = request.form.get("cmgray_smooth")
    
    f_gradient = request.form.get("gradient")
    f_shuffle = True if request.form.get("shuffle") == "true" else False
    
    selectedview = request.form.get("selectedview")
    f_centerpoint = request.form.get("centerpoint")
    
    width = None
    try:
        width = abs(float(request.form.get("width"))) # has to be positive
        width = width if width > 0 else Viewport.default_presets.get(selectedview, "entire-fractal").width # and > 0 
    except ValueError as e:
        # try default
        print(f"Error: {e} \n Setting default value")
        width = Viewport.default_presets.get(selectedview, "entire-fractal").width
    print(f"width={width}")
    
    max_iterations = None
    try:
        max_iterations = abs(int(request.form.get("iterations")))  # has to be positive
        max_iterations = max_iterations if max_iterations > 0 else Viewport.default_presets.get(selectedview, "entire-fractal").max_iterations # and > 0 
        # Safe guard max_iterations
        if max_iterations > MAX_MANDELBROT_ITERATIONS:
            max_iterations = MAX_MANDELBROT_ITERATIONS
    except ValueError as e:
        # try default
        print(f"Error: {e} \n Setting default value")
        max_iterations = Viewport.default_presets.get(selectedview, "entire-fractal").max_iterations
    print(f"max_iterations={max_iterations}")
    
    my_preset = Preset(center=complex(f_centerpoint), width=width, max_iterations=max_iterations, name=selectedview, display_name="User Defined")
    
    print(f"cmap : {colormap_name}")
    print(f"cmexterior : {cmexterior}")
    print(f"cminterior : {cminterior}")
    print(f"cmgray : {cmgray}")
    print(f"f_shuffle : {f_shuffle}")
    
    # build your own color map/palette
    colors = None
    if colormap_name == "manual":
        """
        Choosing a round number for your palette, such as 100 colors, will simplify the formulas. 
        Then, you can split the colors so that 50% goes to the exterior, 5% to the interior, 
        and the remaining 45% to the gray area in between. You want both the exterior and interior 
        to remain white by setting their RGB channels to fully saturated. However, the middle ground 
        should gradually fade from white to black.
        """
        n_of_colors = 100
        ext_color = hex_to_normalized(cmexterior)
        ext_colors = build_color_array(ext_color, cmexterior_smooth, max(1, round(n_of_colors*0.5)))
        if f_shuffle:
            shuffle(ext_colors)
                    
        int_color = hex_to_normalized(cminterior)
        int_colors = build_color_array(int_color, cminterior_smooth, max(1, round(n_of_colors*0.05)))
        # print(f"int_colors:{int_colors}")
        if f_shuffle:
            shuffle(int_colors)
        # print(f"int_colors:{int_colors}")
        
        gray_color = hex_to_normalized(cmgray)
        gray_colors = build_color_array(gray_color, cmgray_smooth, max(1, round(n_of_colors*0.45)))
        if f_shuffle:
            shuffle(gray_colors)
        
        colors = ext_colors + gray_colors + int_colors
        if f_gradient in ["linear", "cubic"]:
            gradient = Helper.make_gradient(colors, interpolation=f_gradient)
            num_colors = 256
            colors = [gradient(i / num_colors) for i in range(num_colors)]
    else:
        colors = np.array(matplotlib.colormaps[colormap_name].colors)
        # print(f"180-colors:{colors}")
        if f_shuffle:
            shuffle(colors)
        if f_gradient in ["linear", "cubic"]:
            gradient = Helper.make_gradient(colors, interpolation=f_gradient)
            num_colors = 256
            colors = [gradient(i / num_colors) for i in range(num_colors)]
            
    palette = Helper.denormalize(colors)
        
    image = generate_mandelbrot2(imgtype, palette, preset=my_preset)
    if CODEHARMONY:
        image = codeharmony(image)
    image_buffer = BytesIO()
    image.save(image_buffer, "PNG")
    image_buffer.seek(0)
    encoded_image = b64encode(image_buffer.getbuffer()).decode("utf-8")
    json_response["image"] = encoded_image
    return json.dumps(json_response)

def build_color_array(base_color, smoothness, count):
    delta = 0.005
    p = []
    if smoothness == "single":
        p = [base_color] * count
    if smoothness == "shades":
        p = generate_shades(base_color, count, delta)
    if smoothness == "complementary":
        # Calculate the complementary color by subtracting each component from 1
        complementary_color = tuple(1 - x for x in base_color)
        half_size = max(1, count // 2) # make sure size is not 0
        p = generate_shades(base_color, half_size, delta)
        p = p + generate_shades(complementary_color, half_size, delta)
    
    # shuffle(p)
    return p

def hex_to_normalized(rgb_hex):
    # Remove the '#' if present
    rgb_hex = rgb_hex.lstrip("#")

    # Parse the hex values for red, green, and blue
    r = int(rgb_hex[0:2], 16) / 255.0
    g = int(rgb_hex[2:4], 16) / 255.0
    b = int(rgb_hex[4:6], 16) / 255.0

    # Return the normalized values
    return (r, g, b)

def generate_shades(base_color, num_shades, delta ):
    shades = []
    for i in range(num_shades):
        shade = tuple(np.clip(np.array(base_color) - i * delta, 0, 1))
        shades.append(shade)
    # reverse the array so darker to lighter 
    shades = shades[::-1]
    return shades
    
def generate_mandelbrot1(width, height, max_iter=1000):
    # Create a blank Pixmap
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height))

    # Define the region of the complex plane to visualize
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5

    # Generate the Mandelbrot set
    for x in range(width):
        for y in range(height):
            zx, zy = x * (xmax - xmin) / (width - 1) + xmin, y * (ymax - ymin) / (height - 1) + ymin
            c = zx + zy * 1j
            z = c
            for i in range(max_iter):
                if abs(z) > 2.0:
                    break
                z = z * z + c
            # Set pixel color based on the number of iterations
            pix.set_pixel(x, y, (i % 256, (i * 7) % 256, (i * 13) % 256))

    return pix

def generate_mandelbrot2(imgtype, palette, preset=Viewport.default_presets.get("entire-fractal", None)):
    # preset = Viewport.default_presets.get("entire-fractal", None)
    image = None
    
    if imgtype == IMAGE_COLOR:
        # colormap = matplotlib.colormaps[colormap_name].colors
        # palette = Helper.denormalize(colormap)

        # mandelbrot_set = MandelbrotSet(max_iterations=512, escape_radius=1000)
        mandelbrot_set = MandelbrotSet(max_iterations=preset.max_iterations, escape_radius=1000)
        image = Image.new(mode=IMAGE_COLOR, size=(512, 512))
        # viewport = Viewport(image, center=-0.7435 + 0.1314j, width=0.002)
        viewport = Viewport(image, center=preset.center, width=preset.width)
        Helper.paint(mandelbrot_set, viewport, palette, smooth=True)
    else:
        mandelbrot_set = MandelbrotSet(max_iterations=preset.max_iterations, escape_radius=1000)
        image = Image.new(mode=imgtype, size=(512, 512))
        # for pixel in Viewport(image, center=-0.7435 + 0.1314j, width=0.002):
        for pixel in Viewport(image, center=preset.center, width=preset.width):
            c = complex(pixel)
            instability = 1 - mandelbrot_set.stability(c, smooth=True)
            pixel.color = int(instability * 255)
            
    return image

@image_tools_bp.route('/image')
def image():
    return render_template('image_tools/image.html', default_presets=Viewport.default_presets)

@image_tools_bp.route('/image/process-image', methods=['POST'])
def process_image():
    """Get thumbnail image of uploaded PDF file"""
    json_response = {
        "status": "success",
        "description": "Successfuly completed operation",
        "results": []
    }
    # get uploaded file
    operation_type = request.form.get("operation-type", "")
    files = request.files.getlist("files") 
    
    print("---------------------------------------------")
    for key, value in request.form.items():
        print(f'{key}: {value}')
    print("---------------------------------------------")
    
    if operation_type == "get-palette":
        # operation get-palette input(s)
        fpalette_size = request.form.get("palette-size")
        palette_size = DEFAULT_COLOR_PALETTE_SIZE
        try:
            palette_size = min(abs(int(fpalette_size)),MAX_COLOR_PALETTE_SIZE)  # has to be positive
        except ValueError as e:
            # try default
            palette_size = DEFAULT_COLOR_PALETTE_SIZE
        # Process all files
        for file in files:
            file_json = {}
            file_json["file_name"] = file.filename
            file_json["content_type"] = file.content_type
            file_json["file_size"] = file.content_length
            file_json["image"] = b64encode(get_palette(file, palette_size)).decode("utf-8")
            file_json["image_attribute"] = 'width="100%"'
            file_json["description"] = f'Color palette of {palette_size} colors from your image'
            json_response["results"].append(file_json)
    
    if operation_type == "create-thumbnail":
        # operation create-thumbnail input(s)
        fthumbnail_width = request.form.get("thumbnail-width", "")
        print(f"fthumbnail_width: {fthumbnail_width}")
        fthumbnail_height = request.form.get("thumbnail-height", "")
        print(f"fthumbnail_height: {fthumbnail_height}")
        
        thumbnail_width = -1
        try:
            thumbnail_width = int(fthumbnail_width) 
        except ValueError as e:
            # try default
            thumbnail_width = -1
        
        thumbnail_height = -1
        try:
            thumbnail_height = int(fthumbnail_height) 
        except ValueError as e:
            # try default
            thumbnail_height = -1
        
        # Process all files
        for file in files:
            file_json = {}
            file_json["file_name"] = file.filename
            file_json["content_type"] = file.content_type
            file_json["file_size"] = file.content_length
            file_json["image"] = b64encode(create_thumbnail(file, thumbnail_width, thumbnail_height)).decode("utf-8")
            file_json["description"] = f'{thumbnail_width}x{thumbnail_height} thumbnail of your image'
            json_response["results"].append(file_json)
    
    if operation_type == "resize":
        # operation resize input(s)
        fresize_width = request.form.get("resize-width", "")
        print(f"fresize_width: {fresize_width}")
        fresize_height = request.form.get("resize-height", "")
        print(f"fresize_height: {fresize_height}")
        
        resize_width = -1
        try:
            resize_width = int(fresize_width) 
        except ValueError as e:
            # try default
            resize_width = -1
        
        resize_height = -1
        try:
            resize_height = int(fresize_height) 
        except ValueError as e:
            # try default
            resize_height = -1
        
        # Process all files
        for file in files:
            file_json = {}
            file_json["file_name"] = file.filename
            file_json["content_type"] = file.content_type
            file_json["file_size"] = file.content_length
            file_json["image"] = b64encode(resize(file, resize_width, resize_height)).decode("utf-8")
            file_json["description"] = f'resized your image to {resize_width}x{resize_height}'
            json_response["results"].append(file_json)
            
    return json_response

def resize(file, width, height):
    """
    Resize image from given image file
    if width,height is <=0 then image 50% scaled image is returned
    if only one of them provided the other is calculated to same scale

    Parameters:
    - file (FileStorage): FileStorage object containing image 
    - width (int): Scaled image width
    - height (int): Scaled image height
    
    Returns:
    memoryview: byte data from BytesIO().getbuffer()
    """
    image_file_stream = BytesIO(file.read())
    image = Image.open(image_file_stream)
    
    # size = width, height
    if width <= 0 and height <= 0:
        width,height = image.width//2, image.height//2 # use default
    if height <= 0 and width > 0:
        height = int(image.height * (width/image.width))
    if width <= 0 and height > 0:
        width = int(image.width * (height/image.height))
    
    size = width, height
    print(f"image (WxH): {width}x{height}")
    # print(f"image (WxH): {image.width}x{image.height}")
    
    image = image.resize(size, reducing_gap=2.0)
    image_file_stream = BytesIO()
    image.save(image_file_stream, format="PNG")
    image_file_stream.seek(0)
    return image_file_stream.getbuffer()

def create_thumbnail(file, width, height):
    """
    Create thumbail images from given image file
    if width,height is <=0 then default (32,32) is used
    if only one of them provided the other is calculated to same scale

    Parameters:
    - file (FileStorage): FileStorage object containing image 
    - width (int): Thumbnail width
    - height (int): Thumbnail height
    
    Returns:
    memoryview: byte data from BytesIO().getbuffer()
    """
    image_file_stream = BytesIO(file.read())
    image = Image.open(image_file_stream)
    
    # size = width, height
    if width <= 0 and height <= 0:
        width,height = 32, 32 # use default
    if width <= 0 and height > 0:
        width = image.width * (height/image.height)
    
    size = width, height
    # print(f"thumbnail (WxH): {width}x{height}")
    # print(f"image (WxH): {image.width}x{image.height}")
    
    image.thumbnail(size)
    image_file_stream = BytesIO()
    image.save(image_file_stream, format="PNG")
    image_file_stream.seek(0)
    return image_file_stream.getbuffer()

def get_palette(file, palette_size):
    image_file_stream = BytesIO(file.read())
    image = Image.open(image_file_stream)
    # Convert the image to RGBA mode
    # image = image.convert('RGBA')
    
    image = image.convert("P", palette = Image.ADAPTIVE, colors = palette_size)
    palette = np.array(image.getpalette()).reshape(palette_size,3)
    
    plt.cla()
    plt.clf()
    
    # Use color strip for up to DEFAULT_COLOR_PALETTE_SIZE
    if palette_size <= DEFAULT_COLOR_PALETTE_SIZE:
        labels = ["#{:02X}{:02X}{:02X}".format(*c) for c in palette]
        sns.palplot(sns.color_palette(labels))
        
        ## Use Seaborn instead
        # new_palette =  [tuple(row) for row in palette]
        # new_palette = np.array(new_palette)[np.newaxis, :, :]
        # plt.imshow(new_palette)
        
        plt.axis('off')
        # plt.title(f'Color palette of {palette_size} colors from your image')
        plot_stream = BytesIO()
        plt.savefig(plot_stream)
        return plot_stream.getbuffer()
    # else Use treemap
    else:
        # Convert color codes to RGB format for plotting
        rgb_colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in palette]
        # Sample data (sizes of each category)
        sizes = np.ones(len(rgb_colors))
        # Labels for each category
        labels = ["#{:02X}{:02X}{:02X}".format(*c) for c in palette]
        
        # Create a treemap-like representation
        plt.figure(figsize=(8, 8))
        squarify.plot(sizes, label=labels, color=rgb_colors, alpha=0.7)
        # Add labels and customize the plot
        # plt.title(f'Top {palette_size} Most Common Colors in Image')
        plt.axis('off')  # Turn off axis labels

        # Save the plot
        plot_stream = BytesIO()
        plt.savefig(plot_stream)
        return plot_stream.getbuffer()
    
