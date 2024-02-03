# viewport.py

from dataclasses import dataclass
from PIL import Image

@dataclass
class Preset:
    center: complex
    width: float
    max_iterations: int
    name: str  
    display_name: str  

@dataclass
class Viewport:
    image: Image.Image
    center: complex
    width: float
    
    default_presets = {
        "entire-fractal" : Preset(center=complex(-0.75, 0), width=3.5, max_iterations=20, name="entire-fractal", display_name="Entire Fractal"),
        "misiurewicz-point" : Preset(center=complex(-0.7435, 0.1314), width=0.002, max_iterations=1000, name="misiurewicz-point", display_name="Misiurewicz Point"),
        "seahorse-valley" : Preset(center=complex(-0.75, 0.11), width=0.2, max_iterations=200, name="seahorse-valley", display_name="Seahorse Valley"),
        "elephant-valley" : Preset(center=complex(0.33, 0.01), width=0.2, max_iterations=200, name="elephant-valley", display_name="Elephant Valley"),
        "antenna" : Preset(center=complex(-1.34, 0.0), width=0.25, max_iterations=200, name="antenna", display_name="Antenna"),
        "top-antenna" : Preset(center=complex(-0.109, 0.89), width=0.02, max_iterations=400, name="top-antenna", display_name="Top Antenna"),
        "hypnotize" : Preset(center=complex(-0.34842633784126914, -0.60653940234393235), width=0.00022, max_iterations=2000, name="hypnotize", display_name="Hypnotize")
    }
    
    @property
    def height(self):
        return self.scale * self.image.height

    @property
    def offset(self):
        return self.center + complex(-self.width, self.height) / 2

    @property
    def scale(self):
        return self.width / self.image.width

    def __iter__(self):
        for y in range(self.image.height):
            for x in range(self.image.width):
                yield Pixel(self, x, y)
                

@dataclass
class Pixel:
    viewport: Viewport
    x: int
    y: int

    @property
    def color(self):
        return self.viewport.image.getpixel((self.x, self.y))

    @color.setter
    def color(self, value):
        self.viewport.image.putpixel((self.x, self.y), value)

    def __complex__(self):
        return (
                complex(self.x, -self.y)
                * self.viewport.scale
                + self.viewport.offset
        )
        
    