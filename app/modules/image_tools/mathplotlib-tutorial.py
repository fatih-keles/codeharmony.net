#%%

# import numpy as np
# import fitz
# import math
# #==============================================================================
# # create a fun-colored width * height PNG with fitz and numpy
# #==============================================================================
# height = 150
# width  = 100
# bild = np.ndarray((height, width, 3), dtype=np.uint8)

# for i in range(height):
#     for j in range(width):
#         # one pixel (some fun coloring)
#         # bild[i, j] = [(i+j)%256, i%256, j%256]
#         bild[i, j] = [math.log(i+j)%256, i%256, j%256]

# samples = bytearray(bild.tobytes())    # get plain pixel data from numpy array
# pix = fitz.Pixmap(fitz.csRGB, width, height, samples, False)
# pix.save("/tmp/test.png")


# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
# matplotlib.use("Agg")

xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

# %%
plt.plot(xpoints, ypoints)
# plt.show()

# %%
plt.plot(xpoints, ypoints, 'o')
# plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.plot(xpoints, ypoints)

# %%
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o')

# %%
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, 'o:r')

# %%
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, linestyle = 'dotted')

# %%
import matplotlib.pyplot as plt
import numpy as np

y1 = np.array([3, 8, 1, 10])
y2 = np.array([6, 2, 7, 11])

plt.plot(y1)
plt.plot(y2)

# %%
import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1, x2, y2)

# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.grid()
plt.plot(x, y)

# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.plot(x, y)

# %%
import matplotlib.pyplot as plt
import numpy as np

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

#the figure has 1 row, 2 columns, and this plot is the first plot.
plt.subplot(1, 2, 1)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

#the figure has 1 row, 2 columns, and this plot is the second plot.
plt.subplot(1, 2, 2)
plt.plot(x,y)

# %% 
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)

# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y, color = 'hotpink')

x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y, color = '#88c999')

# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

plt.scatter(x, y, c=colors, cmap='viridis')

plt.colorbar()

# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(100, size=(100))
y = np.random.randint(100, size=(100))
colors = np.random.randint(100, size=(100))
sizes = 10 * np.random.randint(100, size=(100))

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')

plt.colorbar()

# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y)

# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(170, 10, 250)

plt.hist(x)

# %%
import matplotlib.pyplot as plt
import numpy as np

y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
myexplode = [0.2, 0, 0, 0]

plt.pie(y, labels = mylabels, explode = myexplode)

# %%
import numpy as np

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
# plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
# plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.title(r'$\sigma_i=15$')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
# plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot()

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.ylim(-2, 2)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the open interval (0, 1)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthresh=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

# plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# some random data
x = np.random.randn(1000)
y = np.random.randn(1000)


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')


# %%
#
# Defining the axes positions using a gridspec
# --------------------------------------------
#
# We define a gridspec with unequal width- and height-ratios to achieve desired
# layout.  Also see the :ref:`arranging_axes` tutorial.

# Start with a square Figure.
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(x, y, ax, ax_histx, ax_histy)


# %%
#
# Defining the axes positions using inset_axes
# --------------------------------------------
#
# `~.Axes.inset_axes` can be used to position marginals *outside* the main
# axes.  The advantage of doing so is that the aspect ratio of the main axes
# can be fixed, and the marginals will always be drawn relative to the position
# of the axes.

# Create a Figure, which doesn't have to be square.
fig = plt.figure(layout='constrained')
# Create the main axes, leaving 25% of the figure space at the top and on the
# right to position marginals.
ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
# The main axes' aspect can be fixed.
ax.set(aspect=1)
# Create marginal axes, which have 25% of the size of the main axes.  Note that
# the inset axes are positioned *outside* (on the right and the top) of the
# main axes, by specifying axes coordinates greater than 1.  Axes coordinates
# less than 0 would likewise specify positions on the left and the bottom of
# the main axes.
ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(x, y, ax, ax_histx, ax_histy)

plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = np.asarray(Image.open('/tmp/stinkbug.png'))
# Clear alpa
img = img[:,:,:3]
# extract color channels 
r = img[:,:,0].ravel()
g = img[:,:,1].ravel()
b = img[:,:,2].ravel()

plt.hist(b, 50, density=True, facecolor='g', alpha=0.75)

# %%
from PIL import Image
import numpy as np
from collections import Counter
# from math import sqrt
import matplotlib.pyplot as plt

def color_distance(color1, color2):
    # return sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))
    return np.sqrt(np.sum((np.array(color1, dtype=np.float32) - np.array(color2, dtype=np.float32)) ** 2))

def merge_similar_colors(counter, threshold, most_common=-1):
    # first reduce by eliminating low frequency 
    print(f"Before: {len(counter.keys())}")
    merged_counter1 = Counter()
    for current_color in counter.keys():
        # current_color = colors.pop(0)
        current_count = counter[current_color]
        
        if current_count > 350: 
            merged_counter1[current_color] = current_count

    print(f"After: {len(merged_counter1.keys())}")

    merged_counter2 = Counter()

    colors = None
    # if most_common <= 0:
    #     colors = list(counter.keys())
    # else:
    #     colors = [item[0] for item in counter.most_common(most_common*10)]
    
    colors = list(merged_counter1.keys())
    while colors:
        current_color = colors.pop(0)
        # current_count = counter[current_color]
        current_count = merged_counter1[current_color]

        # Find similar colors and merge them
        # similar_colors = [color for color in colors if color_distance(current_color, color) < threshold]
        similar_colors = [color for color in colors if color_distance(current_color, color) < threshold and color != current_color]
        for similar_color in similar_colors:
            # current_count += counter[similar_color]
            current_count += merged_counter1[similar_color]
            colors.remove(similar_color)

        # merged_counter[current_color] = current_count
        merged_counter2[current_color] = current_count

    return merged_counter2

# %%
# Open an image
image_path = "/tmp/Klimt_-_Der_Kuss.jpeg" #'/tmp/pi8-plasma.png'
image = Image.open(image_path)
print(f"Image Shape: {image.size}")

fig, ax = plt.subplots()
imgplot = ax.imshow(image)
ax.axis('off')  # Hide the axes

# Convert the image to a NumPy array
image_array = np.array(image)
print(f"Image Shape: {image_array.shape}")
print(f"{image_array[0]}")

# Clear Alpha channel then 
# Reshape the array to a 2D array 
# (rows are pixels, columns are RGB values)
pixels = image_array[:,:,:3].reshape((-1, 3))
print(f"Pixels shape: {pixels.shape}")
# pixels = pixels[:40000,:]
print(f"Pixels shape: {pixels.shape}")
print(pixels[0])

color_counts = Counter(list(map(tuple, pixels)))
# print(sorted(color_counts))
print(f"Before reducing:\n {color_counts.most_common(256)}")

threshold_distance = 10  # Adjust the threshold as needed

reduced_color_counts = merge_similar_colors(color_counts, threshold_distance, most_common=10000)
print(f"After reducing: \n {reduced_color_counts.most_common(256)}")

# for color, count in color_counts.items():
#     print(f"Color: {color}, Occurrences: {count}")

# Print the unique colors
# for color in unique_colors:
#     print(color)
#%%
most_common_colors = reduced_color_counts.most_common(30)
# Extract color codes and occurrences
colors, occurrences = zip(*most_common_colors)
my_colors = np.array([(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors])
sorted_indices = np.lexsort(np.array(my_colors).T)
sorted_rgb_colors = my_colors[sorted_indices]

my_labels = None #["#{:02X}{:02X}{:02X}".format(*c) for c in colors]
plt.pie(np.ones(len(occurrences)), labels = my_labels, colors=sorted_rgb_colors)

# %%
# Get the most common colors and their occurrences
most_common_colors = reduced_color_counts.most_common(256)
# print(f"{most_common_colors}")

# Extract color codes and occurrences
colors, occurrences = zip(*most_common_colors)
# occurrences = [o//1000 for o in occurrences]

# Convert color codes to RGB format for plotting
rgb_colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]

# Plot the bar chart
fig, ax = plt.subplots()
# bars = ax.bar(range(len(occurrences)), occurrences, color=rgb_colors)
bars = ax.bar(range(len(occurrences)), np.ones(len(occurrences)), color=rgb_colors)

# Set x-axis labels to be the color codes
ax.set_xticks(range(len(occurrences)))
# ax.set_xticklabels([f"{c}" for c in colors])
ax.set_xticklabels(["#{:02X}{:02X}{:02X}".format(*c) for c in colors], rotation='vertical')
plt.yscale('log')
plt.grid(visible=True, axis='y')


# Set labels and title
ax.set_xlabel('Color Code')
ax.set_ylabel('Occurrences')
ax.set_title('Top 10 Most Common Colors in Image')

# Add color legend
# ax.legend(bars, rgb_colors, loc='upper right')

# Show the plot
plt.show()

# %%
import math

target_distance = 10

sol = []
for a in range(0, target_distance+1):
    for b in range(0, target_distance+1):
        for c in range(0, target_distance+1):
            distance = math.sqrt(a**2 + b**2 + c**2)
            #if math.isclose(distance, target_distance):
            if distance <= target_distance:
                sol.append((a, b, c))
                # print(f"Solution found: (a, b, c) = ({a}, {b}, {c}) for {distance}")
print(f"{len(sol)} items")

# %%
import matplotlib.pyplot as plt
import squarify  # You may need to install squarify using: pip install squarify

most_common_colors = reduced_color_counts.most_common(10)
print(f"{most_common_colors}")

# Extract color codes and occurrences
colors, occurrences = zip(*most_common_colors)
# Convert color codes to RGB format for plotting
rgb_colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]

# Sample data (sizes of each category)
sizes = occurrences #[50, 30, 20]

# Labels for each category
labels = ["#{:02X}{:02X}{:02X}".format(*c)+"-"+str(o//1000)+"K" for c,o in most_common_colors] #['Category A', 'Category B', 'Category C']

# Colors for each category
colors = rgb_colors #['#ff9999', '#66b3ff', '#99ff99']

# Create a treemap-like representation
plt.figure(figsize=(8, 8))
squarify.plot(sizes, label=labels, color=colors, alpha=0.7)

# Add labels and customize the plot
plt.title('Top 10 Most Common Colors in Image')
plt.axis('off')  # Turn off axis labels

# Show the plot
plt.show()

# %%
from PIL import Image
import numpy as np
from collections import Counter
# from math import sqrt
import matplotlib.pyplot as plt

image_path = "/tmp/Klimt_-_Der_Kuss.jpeg" #'/tmp/pi8-plasma.png'
image = Image.open(image_path)
print(f"Image Shape: {image.size}")
image_array = np.array(image)
print(image_array.ndim)
# print(image_array)

colors = np.concatenate(image_array, axis=0)
np.sort(colors)
image_array = image = None
print(colors.base)
color_counts = Counter(list(map(tuple, colors)))
# print(color_counts)

# %%
def drawPlot(most_common_colors):
    # Extract color codes and occurrences
    colors, occurrences = zip(*most_common_colors)
    # occurrences = [o//1000 for o in occurrences]

    # Convert color codes to RGB format for plotting
    rgb_colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]

    # Plot the bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(occurrences)), occurrences, color=rgb_colors)

    # Set x-axis labels to be the color codes
    ax.set_xticks(range(len(occurrences)))
    # ax.set_xticklabels([f"{c}" for c in colors])
    ax.set_xticklabels(["#{:02X}{:02X}{:02X}".format(*c) for c in colors], rotation='vertical')
    plt.yscale('log')
    plt.grid(visible=True, axis='y')


    # Set labels and title
    ax.set_xlabel('Color Code')
    ax.set_ylabel('Occurrences')
    ax.set_title('Top 10 Most Common Colors in Image')

    # Add color legend
    # ax.legend(bars, rgb_colors, loc='upper right')

    # Show the plot
    plt.show()
# %%
def drawHistogram(x_data, color, title):
    n, bins, patches = plt.hist(x_data, 50, density=True, facecolor=color, alpha=0.75)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    # plt.title(r'$\sigma_i=15$')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()


# %%
# print(color_counts.most_common(100))
# drawPlot(color_counts.most_common(10))
# print(colors)

print(f"Before: {len(color_counts.keys())}")

merged_counter = Counter()
colors = None
# colors = list(color_counts.keys())
for current_color in color_counts.keys():
    # current_color = colors.pop(0)
    current_count = color_counts[current_color]
    
    if current_count > 350: 
        merged_counter[current_color] = current_count

print(f"After: {len(merged_counter.keys())}")
        
#     # Find similar colors and merge them
#     similar_colors = [color for color in colors if color_distance(current_color, color) < threshold]
#     for similar_color in similar_colors:
#         current_count += counter[similar_color]
#         colors.remove(similar_color)

#     merged_counter[current_color] = current_count

# colors = [item[0] for item in counter.most_common(most_common*10)]
# counter[current_color]

# unique_values, counts = np.unique(colors, return_counts=True)
# print(f"unique_values: {unique_values}")
# print(f"counts: {counts}")

# rgb_loop = [
#     {'index':0,'code':'r', 'name':'Red'},
#     {'index':1,'code':'g', 'name':'Green'},
#     {'index':2,'code':'b', 'name':'Blue'}]

# for c in rgb_loop:
#     print(f"{c['name']}")
#     c_arr = colors[:,c['index']]
#     # print(f"len: {len(c_arr)}")
#     # print(f"min: {np.min(c_arr)}")
#     # print(f"max: {np.max(c_arr)}")
#     # print(f"mean: {np.mean(c_arr)}")
#     # print(f"median: {np.median(c_arr)}")
#     # print(f"std: {np.std(c_arr)}")
    
#     unique_values, counts = np.unique(c_arr, return_counts=True)
#     print("Unique values:", len(unique_values))
#     print("Counts:", counts)

# drawHistogram(colors[:,0], 'r', 'Red')
# drawHistogram(colors[:,1], 'g', 'Green')
# drawHistogram(colors[:,2], 'b', 'Blue')

# %%
import numpy as np

# Example array with duplicate values
arr = np.array([1, 2, 3, 1, 2, 4, 5, 3, 6, 7, 7])

# Get unique values
unique_values, counts = np.unique(arr, return_counts=True)
print("Unique values:", unique_values)
print("Counts:", counts)
# list(map(tuple, (unique_values, counts)))

# %%
import math

a = np.array(([149, 120,  78],
       [147, 119,  79],
       [143, 118,  77]))
b = [round(math.sqrt((i[0]**2)+(i[1]**2)+(i[2]**2))) for i in a]
print(b)
c = np.c_[a, b]
print(f"{c}")
print(f"{c[:,:3]}")

# %%
# https://www.alanzucconi.com/2015/09/30/colour-sorting/
import random

colours_length = 1000
colours = []
for i in range(0, colours_length):
    data = [
            random.random(),
            random.random(),
            random.random()
        ]
    colours.append(data)
# print(colours)
colours.sort()

from PIL import Image

img_data = np.array(colours)
img_data = np.stack((img_data, img_data, img_data, img_data, img_data), axis=0)
print(img_data.shape)
image = Image.frombytes('RGB', (1000, 5), img_data, 'raw')
image.show()
# %%
image_path = "/tmp/Klimt_-_Der_Kuss.jpeg" #'/tmp/pi8-plasma.png'
image = Image.open(image_path)

#
exif = image.getexif()
for k, v in exif.items():
  print(f"Tag: {k} Value: {v}")  # Tag 274 Value 2 

# from PIL import ExifTags
# gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
# print(exif[ExifTags.Base.Software])  # PIL
# print(gps_ifd[ExifTags.GPS.GPSDateStamp])  # 1999:99:99 99:99:99
# print(gps_ifd)

palette_size = 56
# image = image.convert()
image = image.convert("P", palette = Image.ADAPTIVE, colors = palette_size)
# print(f"img_colors:{image.getcolors(maxcolors=256)}")
palette = np.array(image.getpalette()).reshape(palette_size,3)
# print(palette)

# print(len(image.getpalette()))
# Get the most common colors and their occurrences
most_common_colors = palette
# print(f"{most_common_colors}")

# Extract color codes and occurrences
colors, occurrences = palette, np.ones(len(palette))
# occurrences = [o//1000 for o in occurrences]

# Convert color codes to RGB format for plotting
rgb_colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]

# Plot the bar chart
fig, ax = plt.subplots()
# bars = ax.bar(range(len(occurrences)), occurrences, color=rgb_colors)
bars = ax.bar(range(len(occurrences)), np.ones(len(occurrences)), color=rgb_colors)

# Set x-axis labels to be the color codes
ax.set_xticks(range(len(occurrences)))
# ax.set_xticklabels([f"{c}" for c in colors])
ax.set_xticklabels(["#{:02X}{:02X}{:02X}".format(*c) for c in colors], rotation='vertical')
plt.yscale('log')
plt.grid(visible=True, axis='y')


# Set labels and title
ax.set_xlabel('Color Code')
ax.set_ylabel('Occurrences')
ax.set_title('Top 10 Most Common Colors in Image')

# Add color legend
# ax.legend(bars, rgb_colors, loc='upper right')

# Show the plot
plt.show()

# %%
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

image_path = "/tmp/Klimt_-_Der_Kuss.jpeg" #'/tmp/pi8-plasma.png'
image = Image.open(image_path)
image_array = np.array(image)

# display the image
fig, ax = plt.subplots()
plt.imshow(image)

# tuple to select colors of each channel line
colors = ("red", "green", "blue")

# create the histogram plot, with three lines, one for
# each color
plt.figure()
plt.xlim([0, 256])
for channel_id, color in enumerate(colors):
    histogram, bin_edges = np.histogram(
        image_array[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=color)

plt.title("Color Histogram")
plt.xlabel("Color value")
plt.ylabel("Pixel count")

# %%
from PIL import Image
import matplotlib.pyplot as plt
import squarify 
import numpy as np
import seaborn as sns

image_path = "/tmp/Klimt_-_Der_Kuss.jpeg" #'/tmp/pi8-plasma.png'
image = Image.open(image_path)
image_array = np.array(image)

# display the image
# fig, ax = plt.subplots()
# plt.imshow(image)

# Convert image to access palette
palette_size = 16
image = image.convert("P", palette = Image.ADAPTIVE, colors = palette_size)
palette = np.array(image.getpalette()).reshape(palette_size,3)
# print(palette)
# plt.imshow(image)

# Extract color codes and occurrences
# colors, occurrences = zip(*palette)
# Convert color codes to RGB format for plotting
rgb_colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in palette]

# Sample data (sizes of each category)
sizes = np.ones(len(rgb_colors)) #[50, 30, 20]

# Labels for each category
labels = ["#{:02X}{:02X}{:02X}".format(*c) for c in palette] #['Category A', 'Category B', 'Category C']

# Colors for each category
colors = rgb_colors #['#ff9999', '#66b3ff', '#99ff99']

# # Create a treemap-like representation
# plt.figure(figsize=(8, 8))
# squarify.plot(sizes, label=labels, color=colors, alpha=0.7)

# # Add labels and customize the plot
# plt.title('Top 10 Most Common Colors in Image')
# plt.axis('off')  # Turn off axis labels

# # Show the plot
# plt.show()

# Using Seaborn 
gyr = ['#28B463','#FBFF00', '#C0392B']
sns.palplot(sns.color_palette(labels))

# Matplotlib 
# new_palette =  [(c[0], c[1], c[2]) for c in palette]
# new_palette = [tuple(row) for row in palette]
# print(f"<<<<<<{new_palette}")
# new_palette = np.array(new_palette)[np.newaxis, :, :]
# print(f">>>>>{new_palette}")
# # palette2 = [(245, 213, 77), (240, 187, 58), (211, 180, 90), (180, 140, 67), (116, 126, 76), (132, 101, 58), (100, 96, 56), (101, 76, 46), (61, 75, 49), (38, 35, 21)]#[(82, 129, 169), (218, 223, 224), (147, 172, 193), (168, 197, 215), (117, 170, 212)]

# # palette2 = np.array(new_palette)[np.newaxis, :, :]

# plt.imshow(new_palette)
# plt.axis('off')
# plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.randn(1000)

# Create histogram
hist, bin_edges = np.histogram(data, bins=30, density=True)

# Plot the histogram
plt.hist(data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
# %%
from PIL import Image
import matplotlib.pyplot as plt

with Image.open("/tmp/Klimt_-_Der_Kuss.jpeg") as im:
    print(f"({im.width}x{im.height})")
    # Provide the target width and height of the image
    (width, height) = (im.width // 2, im.height // 2)
    im_resized = im.resize((4506, 9000))
    print(f"({im_resized.width}x{im_resized.height})")
    plt.imshow(im_resized)