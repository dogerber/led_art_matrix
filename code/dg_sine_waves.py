#!/usr/bin/env python
from samplebase import SampleBase
import numpy as np
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
        

        def gaussian_kernel(size, sigma):
            """Generate a 2D Gaussian kernel."""
            kernel = np.fromfunction(
                lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)),
                (size, size)
            )
            return kernel / np.sum(kernel)
        
        def gaussian_blur(matrix, kernel_size=3, sigma=2):
            matrix = matrix.astype(float)
            kernel = gaussian_kernel(kernel_size,sigma)
            if False:
                matrix_out = convolve(matrix, kernel, mode='wrap')
            elif True:
                matrix_out = convolve(matrix, kernel, mode='constant', cval =0)

            return matrix_out
        


        # ----------------------- Programs -------------------------------------------
        while True:

                matrix_shape = (64,64)  # Adjust the size of the matrix as needed
                matrix = np.zeros(matrix_shape)

                wl_i = 11 # wavenumber
                wl_j = 5
                dt_i = 10 # frequency
                dt_j = 30
                shift_i = 1
                shift_j=0
                A_i = 1 # amplitude
                A_j = 2
                t = 0

                for _ in range(10000):

                    t +=1

                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[1]):
                            matrix[i,j] = A_i*np.sin(i/wl_i + t/dt_i + shift_i)+ A_j*np.sin(j/wl_j + t/dt_j + shift_j)


                    
                    matrix_to_display = matrix

                    # convert to rgb
                    matrix_to_display = grayscale_to_rgb_with_colormap(matrix_to_display, colormap='inferno')


                    # display on leds
                    drawMatrix(self,matrix_to_display)
                    offset_canvas = self.matrix.SwapOnVSync(offset_canvas)
                    self.usleep(5 * 1000)





# Main function
if __name__ == "__main__":
    simple_square = SimpleSquare()
    if (not simple_square.process()):
        simple_square.print_help()
