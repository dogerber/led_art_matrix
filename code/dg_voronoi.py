import numpy as np
from scipy.spatial import Voronoi
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import time
from matplotlib import cm
from scipy.ndimage import convolve


# LED matrix parameters
matrix_width = 64
matrix_height = 64
n_pts = 30

# Generate random control points
np.random.seed(42)  # For reproducibility
control_points = np.random.rand(n_pts, 2) * np.array([matrix_width, matrix_height])

# Perform Voronoi tessellation
vor = Voronoi(control_points)

# LED matrix setup
options = RGBMatrixOptions()
options.rows = matrix_height
options.cols = matrix_width
options.chain_length = 1
options.parallel = 1
options.pwm_bits = 6 # default 11, minimum 1, lower should be less flickering
options.pwm_lsb_nanoseconds = 50
# options.limit_refresh_rate_hz = 60
options.gpio_slowdown = 4
options.hardware_mapping = 'adafruit-hat-pwm'  # or 'adafruit-hat'

matrix = RGBMatrix(options=options)

matrix_cache = np.zeros((matrix_height,matrix_width))

# Function to map LED position to its corresponding Voronoi region index
def map_led_to_region(x, y):
    return vor.point_region[np.argmin(np.sum((vor.points - np.array([x, y]))**2, axis=1))]

# Function to set color for an LED based on its Voronoi region index
def set_led_color(x, y, region_index, color):
    if 0 <= x < matrix_width and 0 <= y < matrix_height:
        matrix.SetPixel(x, y, *color)

def grayscale_to_rgb_with_colormap(gray_image, colormap='viridis', normalize_by=None):
    # # Define the colormap
    # cmap = cm.get_cmap(colormap)

    # # Apply the colormap to the grayscale image
    # rgb_image = cmap(gray_image)

    # return rgb_image
    if False:
        gray_image += np.amin(gray_image)

    # scale to [0,1]
    if normalize_by == None:
        normalize_by=np.amax(gray_image)
        if normalize_by ==0:
            normalize_by=1

    gray_image = gray_image/normalize_by


    # apply colormap
    cmap = cm.get_cmap(colormap)
    if colormap=='Greys':
        cmap = cmap.reversed() # reverse
    rgb_image = (cmap(gray_image) * 255).astype(np.uint8)
    return rgb_image

def drawMatrix(matrix,matrix_to_display):
    for i in range(matrix_to_display.shape[0]):
        for j in range(matrix_to_display.shape[1]):
            matrix.SetPixel(i,j,matrix_to_display[i,j,0],matrix_to_display[i,j,1],matrix_to_display[i,j,2])

# Function to compute the area of a polygon given its vertices
def polygon_area(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_blur(matrix, kernel_size=3, sigma=2):

    kernel = gaussian_kernel(kernel_size,sigma)
    if False:
        matrix_out = convolve(matrix, kernel, mode='wrap')
    elif True:
        matrix_out = convolve(matrix, kernel, mode='constant', cval =0)

    return matrix_out


# Loop to continuously update LED colors based on Voronoi regions
try:
    print("started")
    while True:
        

        # move points
        if False:
            step_size = 0.1
            step_chance = 0.1
            for i,_ in enumerate(control_points):
                if np.random.rand(1)<step_chance:
                    control_points[i,:] = control_points[i,:] + np.random.random((2,))*step_size

        # voronoi calculation
        vor = Voronoi(control_points)

        # Compute Voronoi cell areas
        cell_areas = np.zeros(vor.npoints+1)
        southernmost_y = np.zeros(vor.npoints+1)
        for i, region_index in enumerate(vor.regions):
            if len(region_index) > 2: # had -1 not in region_index and 
                vertices = vor.vertices[region_index]
                # Clip polygon to matrix bounds
                vertices[:, 0] = np.clip(vertices[:, 0], 0, matrix_width)
                vertices[:, 1] = np.clip(vertices[:, 1], 0, matrix_height)
                cell_areas[i] = polygon_area(vertices)
                southernmost_y[i] = np.min(vertices[:, 1])

        

        # Normalize cell areas to the range [0, 1]
        if False:
            normalized_areas = (cell_areas - np.min(cell_areas)) / (np.max(cell_areas) - np.min(cell_areas))
        else:
            normalized_areas = cell_areas


        # determine for each pixel in which voronoi region it is
        for i in range(matrix_width):
            for j in range(matrix_height):
                region_index = map_led_to_region(i, j)
                if False: # color by identifer
                    matrix_cache[i,j] = region_index
                elif True: # color by x coordinate of a vertices
                    matrix_cache[i,j] = southernmost_y[region_index]
                else: # color by polygone size
                    matrix_cache[i,j] = normalized_areas[region_index]

        # apply gaussian blur
        if True:
            matrix_cache = gaussian_blur(matrix_cache, kernel_size = 3, sigma = 2)

        # convert to rgb colors
        matrix_to_display = grayscale_to_rgb_with_colormap(matrix_cache)

        # display on leds
        drawMatrix(matrix,matrix_to_display)
        time.sleep(1)
        print("round")

except KeyboardInterrupt:
    matrix.Clear()
