## source: https://realpython.com/mandelbrot-set-python/

import matplotlib
# matplotlib.use("Agg")  # Use a suitable backend, e.g., TkAgg

import matplotlib.pyplot as plt
# plt.ioff()  # Turn off interactive mode
import numpy as np

np.warnings.filterwarnings("ignore")

def z(n, c):
    if n == 0:
        return 0
    else:
        return z(n - 1, c) ** 2 + c
    
def sequence(c):
    z = 0
    while True:
        yield z
        z = z ** 2 + c
        
def sequence(c, z=0):
    while True:
        yield z
        z = z ** 2 + c
        
def mandelbrot(candidate):
    return sequence(z=0, c=candidate)

def julia(candidate, parameter):
    return sequence(z=candidate, c=parameter)


import numpy as np

def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    re = np.linspace(xmin, xmax, int((xmax - xmin) * pixel_density))
    im = np.linspace(ymin, ymax, int((ymax - ymin) * pixel_density))
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j

def is_stable(c, num_iterations):
    z = 0
    for _ in range(num_iterations):
        z = z ** 2 + c
    return abs(z) <= 2

def get_members(c, num_iterations):
    mask = is_stable(c, num_iterations)
    return c[mask]

density = 21
density = 50
density = 256
density = 512
# density = 4096
c = complex_matrix(-2, 0.5, -1.5, 1.5, pixel_density=density)
members = get_members(c, num_iterations=20)

plt.scatter(members.real, members.imag, color="black", marker=",", s=1)
plt.gca().set_aspect("equal")
plt.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig("/tmp/plt"+str(density)+".png")

c = complex_matrix(-2, 0.5, -1.5, 1.5, pixel_density=density)
plt.imshow(is_stable(c, num_iterations=20), cmap="binary")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig("/tmp/plt-cmap-"+str(density)+".png")


# # Create figure and axis
# fig, ax = plt.subplots()

# # Your plotting code goes here
# c = complex_matrix(-2, 0.5, -1.5, 1.5, pixel_density=density)
# ax.imshow(is_stable(c, num_iterations=20), cmap="binary")
# ax.gca().set_aspect("equal")
# # ax.axis("off")
# ax.tight_layout()
# # plt.show()

# # Customize axis appearance
# ax.spines['top'].set_color('black')  # Color of the top spine
# ax.spines['top'].set_linewidth(2)    # Width of the top spine
# ax.spines['right'].set_color('black') # Color of the right spine
# ax.spines['right'].set_linewidth(2)   # Width of the right spine

# ax.spines['bottom'].set_color('black') # Color of the bottom spine
# ax.spines['bottom'].set_linewidth(2)   # Width of the bottom spine
# ax.spines['left'].set_color('black')   # Color of the left spine
# ax.spines['left'].set_linewidth(2)     # Width of the left spine

# ax.tick_params(axis='both', which='both', direction='in', length=6, width=2)

# plt.savefig("/tmp/output_plot_with_axes.png")

# Using Pillow
from PIL import Image, ImageEnhance
c = complex_matrix(-2, 0.5, -1.5, 1.5, pixel_density=512)
image = Image.fromarray(~is_stable(c, num_iterations=20))
image.save("/tmp/pi.png")

# Built-in function
Image.effect_mandelbrot((512, 512), (-3, -2.5, 2, 2.5), 100).save("/tmp/pi2.png")
Image.effect_mandelbrot((512, 512), (-2, 0.5, -1.5, 1.5), 100).save("/tmp/pi3.png")

def is_stable(c, max_iterations):
    z = 0
    for _ in range(max_iterations):
        z = z ** 2 + c
        if abs(z) > 2:
            return False
    return True

## With data class
from mandelbrot import MandelbrotSet
mandelbrot_set = MandelbrotSet(max_iterations=20)

width, height = 512, 512
scale = 0.0075
BLACK_AND_WHITE = "1"

from PIL import Image
image = Image.new(mode=BLACK_AND_WHITE, size=(width, height))
for y in range(height):
    for x in range(width):
        c = scale * complex(x - width / 2, height / 2 - y)
        image.putpixel((x, y), c not in mandelbrot_set)

image.show()
image.save("/tmp/pi4.png")

## test stability
from mandelbrot import MandelbrotSet
mandelbrot_set = MandelbrotSet(max_iterations=30)

mandelbrot_set.escape_count(0.25)
mandelbrot_set.stability(0.25)

mandelbrot_set.escape_count(0.26)
mandelbrot_set.stability(0.26)
0.26 in mandelbrot_set


## test grayscale
from mandelbrot import MandelbrotSet
mandelbrot_set = MandelbrotSet(max_iterations=20)

width, height = 512, 512
scale = 0.0075
GRAYSCALE = "L"

from PIL import Image
image = Image.new(mode=GRAYSCALE, size=(width, height))
for y in range(height):
     for x in range(width):
         c = scale * complex(x - width / 2, height / 2 - y)
         instability = 1 - mandelbrot_set.stability(c)
         image.putpixel((x, y), int(instability * 255))

image.show()

## after adding log and smoothing for fractional max iterations 
from mandelbrot import MandelbrotSet
mandelbrot_set = MandelbrotSet(max_iterations=30)

mandelbrot_set.stability(-1.2039 - 0.1996j, smooth=True)
mandelbrot_set.stability(42, smooth=True)

## test after clamping
from mandelbrot import MandelbrotSet
mandelbrot_set = MandelbrotSet(max_iterations=20, escape_radius=1000)

width, height = 512, 512
scale = 0.0075
GRAYSCALE = "L"

from PIL import Image
image = Image.new(mode=GRAYSCALE, size=(width, height))
for y in range(height):
    for x in range(width):
        c = scale * complex(x - width / 2, height / 2 - y)
        instability = 1 - mandelbrot_set.stability(c, smooth=True)
        image.putpixel((x, y), int(instability * 255))

image.show()
image.save("/tmp/pi5.1.png")

## after adding viewport and pixel data classes 
"""
The viewport spans 0.002 world units and is centered at -0.7435 + 0.1314j, 
which is close to a Misiurewicz point that produces a beautiful spiral. 
Depending on the number of iterations, you’ll get a darker or brighter image 
with a varying degree of detail. You can use Pillow to increase the brightness 
if you feel like it:
from PIL import ImageEnhance
enhancer = ImageEnhance.Brightness(image)
enhancer.enhance(1.25).show()
"""
from PIL import Image
from mandelbrot import MandelbrotSet
from viewport import Viewport

mandelbrot_set = MandelbrotSet(max_iterations=256, escape_radius=1000)

image = Image.new(mode="L", size=(512, 512))
for pixel in Viewport(image, center=-0.7435 + 0.1314j, width=0.002):
    c = complex(pixel)
    instability = 1 - mandelbrot_set.stability(c, smooth=True)
    pixel.color = int(instability * 255)

image.show()
image.save("/tmp/pi6.png")

## Helper 
import matplotlib.cm
from helper import Helper
 
# colormap = matplotlib.cm.get_cmap("twilight").colors
colormap = matplotlib.colormaps["twilight"].colors
palette = Helper.denormalize(colormap)

len(colormap)

colormap[0]
palette[0]

from PIL import Image
from mandelbrot import MandelbrotSet
from viewport import Viewport

mandelbrot_set = MandelbrotSet(max_iterations=512, escape_radius=1000)
image = Image.new(mode="RGB", size=(512, 512))
viewport = Viewport(image, center=-0.7435 + 0.1314j, width=0.002)
Helper.paint(mandelbrot_set, viewport, palette, smooth=True)

# image.show()
image.save("/tmp/pi7.png")

## Color map test
import matplotlib.cm
from helper import Helper
from PIL import Image
from mandelbrot import MandelbrotSet
from viewport import Viewport

## https://matplotlib.org/stable/gallery/color/colormap_reference.html
colormap_name = "hsv"

colormap = matplotlib.colormaps[colormap_name].colors
palette = Helper.denormalize(colormap)

mandelbrot_set = MandelbrotSet(max_iterations=512, escape_radius=1000)
image = Image.new(mode="RGB", size=(512, 512))
viewport = Viewport(image, center=-0.7435 + 0.1314j, width=0.002)
Helper.paint(mandelbrot_set, viewport, palette, smooth=True)

image.save("/tmp/pi8.png")

## test all color maps
import matplotlib.cm
from helper import Helper
from PIL import Image
from mandelbrot import MandelbrotSet
from viewport import Viewport

mandelbrot_set = MandelbrotSet(max_iterations=512, escape_radius=1000)
image = Image.new(mode="RGB", size=(512, 512))
viewport = Viewport(image, center=-0.7435 + 0.1314j, width=0.002)

for colormap_name in matplotlib.colormaps:
    try:
        print(f"{colormap_name}:")
        colormap = matplotlib.colormaps[colormap_name].colors
        palette = Helper.denormalize(colormap)
        Helper.paint(mandelbrot_set, viewport, palette, smooth=True)
        
        image.save("/tmp/pi8-"+colormap_name+".png")
    except Exception as e:
        print("An error occurred:", str(e))


"""
Suppose you wanted to emphasize the fractal’s edge. In such a case, 
you can divide the fractal into three parts and assign different colors to each:
Choosing a round number for your palette, such as 100 colors, will simplify the formulas. 
Then, you can split the colors so that 50% goes to the exterior, 5% to the interior, 
and the remaining 45% to the gray area in between. You want both the exterior and interior 
to remain white by setting their RGB channels to fully saturated. However, the middle ground 
should gradually fade from white to black.

Don’t forget to set the viewport’s center point at -0.75 and its width to 3.5 units to cover 
the entire fractal. At this zoom level, you’ll also need to drop the number of iterations:
"""
import matplotlib.cm
from helper import Helper
from PIL import Image
from mandelbrot import MandelbrotSet
from viewport import Viewport

exterior = [(1, 1, 1)] * 50
# exterior = [(1, 0, 0)] * 50 ## make it red
interior = [(1, 1, 1)] * 5
# interior = [(0, 1, 0)] * 5 ## make it green
gray_area = [(1 - i / 44,) * 3 for i in range(45)]
# gray_area = [(1 - i / 44, 0 + i / 44, 0.05)  for i in range(45)] ## from red to green 
palette = Helper.denormalize(exterior + gray_area + interior)

mandelbrot_set = MandelbrotSet(max_iterations=20, escape_radius=1000)
# mandelbrot_set = MandelbrotSet(max_iterations=512, escape_radius=1000)
image = Image.new(mode="RGB", size=(512, 512))
viewport = Viewport(image, center=-0.75, width=3.5)
# viewport = Viewport(image, center=-0.7435 + 0.1314j, width=0.002)
Helper.paint(mandelbrot_set, viewport, palette, smooth=True)

image.save("/tmp/pi9.png")

## Gradient colors
black = (0, 0, 0)
blue = (0, 0, 1)
maroon = (0.5, 0, 0)
navy = (0, 0, 0.5)
red = (1, 0, 0)

colors = [black, navy, blue, maroon, red, black]
gradient = Helper.make_gradient(colors, interpolation="cubic")

gradient(0.42)

"""
Note that gradient colors, such as black in the example above, 
can repeat and appear in any order. To hook up the gradient to your 
palette-aware painting function, you must decide on the number of 
colors in the corresponding palette and convert the gradient 
function to a fixed-sized list of denormalized tuples:
"""
num_colors = 256
palette = Helper.denormalize([
    gradient(i / num_colors) for i in range(num_colors)
])

len(palette)
palette[127]

mandelbrot_set = MandelbrotSet(max_iterations=20, escape_radius=1000)
image = Image.new(mode="RGB", size=(512, 512))
viewport = Viewport(image, center=-0.75, width=3.5)
Helper.paint(mandelbrot_set, viewport, palette, smooth=True)

# image.show()
image.save("/tmp/pi9.png")

## HSB color model
import matplotlib.cm
from helper import Helper
from PIL import Image
from mandelbrot import MandelbrotSet
from viewport import Viewport

mandelbrot_set = MandelbrotSet(max_iterations=512, escape_radius=1000)
image = Image.new(mode="RGB", size=(512, 512))
for pixel in Viewport(image, center=-1.4011, width=0.005):
    stability = mandelbrot_set.stability(complex(pixel), smooth=True)
    pixel.color = (0, 0, 0) if stability == 1 else Helper.hsb(
        hue_degrees=int(stability * 360),
        saturation=stability,
        brightness=1,
    )

# image.show()
image.save("/tmp/pi10.png")
