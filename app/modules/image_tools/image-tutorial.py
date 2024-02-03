# %%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
img = np.asarray(Image.open('/tmp/stinkbug.png'))
print(repr(img))

# %%
print(img.shape)
imgplot = plt.imshow(img)

# %%
pixels = img[:,:,:3].reshape((-1, 3))
print(repr(img[0]))
print(repr(pixels[0]))
print(img.shape)
print(pixels.shape)
# imgplot2 = plt.imshow(pixels)

# %%
#we can just pick one channel of our data using array slicing
lum_img = img[:, :, 0]
print(repr(lum_img))
print(lum_img.shape)
plt.imshow(lum_img)

# %%
plt.imshow(lum_img, cmap="hot")

# %%
imgplot = plt.imshow(lum_img)
imgplot.set_cmap('nipy_spectral')
plt.colorbar()

# %%
"""Examining a specific data range
Sometimes you want to enhance the contrast in your image, 
or expand the contrast in a particular region while sacrificing 
the detail in colors that don't vary much, or don't matter. 
A good tool to find interesting regions is the histogram. 
To create a histogram of our image data, we use the hist() 
function."""
plt.hist(lum_img.ravel(), bins=range(256), fc='k', ec='k')

# %%
# clip high end (>175)
plt.imshow(lum_img, clim=(0, 175))

# %%
img = Image.open('/tmp/stinkbug.png')
img.thumbnail((64, 64))  # resizes image in-place
imgplot = plt.imshow(img)

# %% 
# better results with interpolation
imgplot = plt.imshow(img, interpolation="bilinear")

# %%
imgplot = plt.imshow(img, interpolation="bicubic")
# %%
import fitz 
from PIL import Image, ImageDraw
from io import BytesIO

def codeharmony(pm):
    height = pm.height
    width = pm.width
    
    text = "codeharmony.net"
    text_position = (2, height-12)
    
    image = Image.open(BytesIO(pm.tobytes()))
    d = ImageDraw.Draw(image)
    d.text(text=text, xy=text_position, fill="white")
    return image
    

bgcolor = (0,0,255)
ir = (0, 0, 200, 200)  
pm = fitz.Pixmap(fitz.csRGB, ir, False)
pm.set_rect(pm.irect, bgcolor)

# pm.save("/tmp/deneme.png") 
# print(f"{pm.tobytes()}")

# image = Image.frombytes(mode="RGB", data=BytesIO(pm.tobytes()))
# image = Image.open(BytesIO(pm.tobytes()))

image = codeharmony(pm)
image.save("/tmp/deneme3.png")
# %%
from PIL import Image,ExifTags
import datetime

im = Image.open("/tmp/deneme3.png")
exif = im.getexif()  # Returns an instance of this class
exif[ExifTags.Base.Software] = 'CodeHarmony using Pillow'
exif[ExifTags.Base.DateTime] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")

for k, v in exif.items():
  print(f"Tag: {k} Value: {v}")  # Tag 274 Value 2
  
exif = im.getexif()
gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
print(f"{gps_ifd}")
print(exif[ExifTags.Base.Software])  # PIL
print(exif[ExifTags.Base.DateTime])  # PIL
# print(gps_ifd[ExifTags.GPS.GPSDateStamp])  # 1999:99:99 99:99:99
# %%
from PIL import Image
import glob, os

size = 32, 32

for infile in glob.glob("/tmp/images/*.png"):
    file, ext = os.path.splitext(infile)
    with Image.open(infile) as im:
        im.thumbnail(size)
        im.save(file + ".32x32.png", "PNG")