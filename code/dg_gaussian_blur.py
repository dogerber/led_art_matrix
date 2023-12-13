#!/usr/bin/env python
from samplebase import SampleBase

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.ndimage import convolve

class SimpleSquare(SampleBase):
    def __init__(self, *args, **kwargs):
        super(SimpleSquare, self).__init__(*args, **kwargs)



    def run(self):
        offset_canvas = self.matrix.CreateFrameCanvas()

        def drawMatrix(self,matrix):
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    offset_canvas.SetPixel(i,j,matrix[i,j,0],matrix[i,j,1],matrix[i,j,2])




        def drawMatrixSmoothTransition(matrix_start,matrix_end,incr_i,increments_total):
            matrix_to_draw = matrix_start+ (matrix_end - matrix_start)*incr_i/increments_total
            return matrix_to_draw.astype(np.uint8)

            
        
        def grayscale_to_rgb_with_colormap(gray_image, colormap='viridis', normalize_by=None):
            """
            Convert a grayscale image to an RGB image with a specified colormap.

            Parameters:
            - gray_image: 2D NumPy array representing the grayscale image.
            - colormap: Name of the colormap (default is 'viridis').

            Returns:
            - rgb_image: RGB image with the specified colormap.
                see here https://matplotlib.org/stable/users/explain/colors/colormaps.html#sequential
            """

            # # Define the colormap
            # cmap = cm.get_cmap(colormap)

            # # Apply the colormap to the grayscale image
            # rgb_image = cmap(gray_image)

            # return rgb_image

            # scale to [0,1]
            if normalize_by == None:
                normalize_by=np.amax(gray_image)

            gray_image = gray_image/normalize_by
            

            # apply colormap
            cmap = cm.get_cmap(colormap)
            if colormap=='Greys':
                cmap = cmap.reversed() # reverse
            rgb_image = (cmap(gray_image) * 255).astype(np.uint8)
            return rgb_image
        
        
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
        
        




        # ----------------------- Program -------------------------------------------
        while True:

            if True: # random position, random color
                matrix_size = 64
                matrix_grayscale = np.random.randint(0,255,size=(matrix_size,matrix_size))*0
                matrix_before = np.zeros((matrix_size,matrix_size,4),dtype=np.uint8)

                for _ in range(1000):
                    matrix_grayscale = gaussian_blur(matrix_grayscale, kernel_size = 11, sigma = 1)
                    # smaller sigma = sharper blur
                    # kernel size should be odd number

                    
                    # add random seeds
                    for _ in range(5):
                        random_coord = np.random.randint(0,offset_canvas.height,2)
                        matrix_grayscale[random_coord[0],random_coord[1]] = np.random.randint(200,255)


                    # convert to rgb
                    matrix_to_display = grayscale_to_rgb_with_colormap(matrix_grayscale, colormap='inferno', normalize_by = 30)

                    # display on leds
                    if True: # instant change
                        drawMatrix(self,matrix_to_display)
                        offset_canvas = self.matrix.SwapOnVSync(offset_canvas)
                        self.usleep(50 * 1000)
                    else:
                        total_increments = 10
                        for i in range(total_increments):
                            drawMatrix(self,drawMatrixSmoothTransition(matrix_before,matrix_to_display,i, total_increments))
                            offset_canvas = self.matrix.SwapOnVSync(offset_canvas)
                            self.usleep(10 * 1000)
                        matrix_before = matrix_to_display





# Main function
if __name__ == "__main__":
    simple_square = SimpleSquare()
    if (not simple_square.process()):
        simple_square.print_help()
