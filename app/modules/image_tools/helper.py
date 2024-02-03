from PIL import Image
from app.modules.image_tools.mandelbrot import MandelbrotSet
from app.modules.image_tools.viewport import Viewport
import numpy as np
from scipy.interpolate import interp1d
from PIL.ImageColor import getrgb

class Helper:
    
    def paint(mandelbrot_set, viewport, palette, smooth):
        for pixel in viewport:
            stability = mandelbrot_set.stability(complex(pixel), smooth)
            index = int(min(stability * len(palette), len(palette) - 1))
            pixel.color = palette[index % len(palette)]
        
    def denormalize(palette):
        return [
            tuple(int(channel * 255) for channel in color)
            for color in palette
        ]
    
    """
    Your new factory function accepts a list of colors defined as triplets of 
    floating-point values and an optional string with the name for the interpolation 
    algorithm exposed by SciPy. The uppercase X variable contains normalized values 
    between zero and one based on the number of colors. The uppercase Y variable holds 
    three sequences of R, G, and B values for each color, and the channels variable has 
    the interpolation functions for each channel.
    
    The library comes with linear, quadratic, and cubic interpolation methods, 
    among a few others. Here’s how you can take advantage of it:
    """
    def make_gradient(colors, interpolation="linear"):
        X = [i / (len(colors) - 1) for i in range(len(colors))]
        Y = [[color[i] for color in colors] for i in range(3)]
        channels = [interp1d(X, y, kind=interpolation) for y in Y]
        return lambda x: [np.clip(channel(x), 0, 1) for channel in channels]
    
    """
    The three HSB coordinates are:
    Hue: The angle measured counterclockwise between 0° and 360°
    Saturation: The radius of the cylinder between 0% and 100%
    Brightness: The height of the cylinder between 0% and 100%
    To use such coordinates in Pillow, you must translate them to a tuple of RGB values 
    in the familiar range of 0 to 255:
    """
    def hsb(hue_degrees: int, saturation: float, brightness: float):
        return getrgb(
            f"hsv({hue_degrees % 360},"
            f"{saturation * 100}%,"
            f"{brightness * 100}%)"
        )