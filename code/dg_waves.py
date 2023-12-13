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

            gray_image -= np.min(gray_image) # make it start at 0

            # scale to [0,1]
            if normalize_by == None:
                if np.amax(gray_image)!=0:
                    normalize_by=np.amax(gray_image)
                else:
                    normalize_by =1

            gray_image = gray_image/normalize_by
            

            # apply colormap
            cmap = cm.get_cmap(colormap)
            if colormap=='Greys':
                cmap = cmap.reversed() # reverse
            rgb_image = (cmap(gray_image) * 255).astype(np.uint8)
            return rgb_image
        

        
        def custom_convolution(matrix,kernel=None):
            if kernel==None:
                kernel = np.ones((3,3))
                kernel[1,1] = 0
            
            return convolve(matrix,kernel,mode='constant',cval=0)
        
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
        





        # ----------------------- Programs -------------------------------------------
        while True:
            if True: # game of live with history
                matrix_size = 64

                matrix_grayscale = np.zeros((matrix_size,matrix_size))
                if False: # start with points in the corner
                    matrix_grayscale[0,0] = 1
                    matrix_grayscale[0,-1] = 1
                    matrix_grayscale[-1,0] = 1
                    matrix_grayscale[-1,-1] = 1                                                
                
                history_level = 40
                matrix_array = np.zeros((history_level,matrix_size,matrix_size))

                activation_matrix = np.zeros((matrix_size,matrix_size))

                 # add random seeds
                if False:
                    for _ in range(1):
                        random_coord = np.random.randint(0,matrix_size-1,2)
                        matrix_grayscale[random_coord[0],random_coord[1]] = 1

                for _ in range(5000):
                    # print(activation_matrix)
                    # print(matrix_grayscale)

                    # add random seeds
                    if np.random.random(1)<0.1:
                        for _ in range(1):
                            random_coord = np.random.randint(0,matrix_size-1,2)
                            matrix_grayscale[random_coord[0],random_coord[1]] = 1

                    # determine number of neighbours
                    if True:
                        kernel = np.ones((3,3))
                        # kernel[1,1] = 0 # do not count yourself             
                    number_of_neighbours = convolve(matrix_grayscale, kernel, mode='constant')


                    # calculate activation matrix
                    activation_factor = 10
                    activation_matrix[matrix_grayscale!=1] += number_of_neighbours[matrix_grayscale!=1]/activation_factor

                    # decay of activation function
                    if True: #decay if no neighbours
                        activation_matrix[number_of_neighbours==0] = 0
                    else:
                        activation_matrix[activation_matrix<0] +=0.1

                    # determine what dies
                    if False: # with number of neighbour strict
                        too_many_neighbours = 5 # [1-8]
                        death_penalty = -100
                        matrix_grayscale[number_of_neighbours>too_many_neighbours] = 0
                        activation_matrix[number_of_neighbours>too_many_neighbours] = death_penalty
                    else:
                        death_factor = 20
                        death_threshold = -6 # -4 has sources, -6 spirals
                        activation_matrix[matrix_grayscale==1] -=  number_of_neighbours[matrix_grayscale==1]/death_factor
                        matrix_grayscale[activation_matrix <death_threshold] =0


                    # determine what comes alive
                    activation_threshold = 2
                    matrix_grayscale[activation_matrix>=activation_threshold] = 1
                    activation_matrix[activation_matrix>=activation_threshold] =0

                    # note down
                    matrix_array[1:] = matrix_array[:-1]
                    matrix_array[0] = matrix_grayscale

                    
                    
                    matrix_summed =np.zeros_like(matrix_grayscale)
                    for i in range(history_level-1,0,-1):
                        matrix_summed[(matrix_array[i,:,:]>0)] = (history_level-i+1)


                    # Gaussian smoothing
                    matrix_summed = gaussian_blur(matrix_summed, kernel_size = 5, sigma = 4)

                    # convert to rgb
                    matrix_to_display = grayscale_to_rgb_with_colormap(matrix_summed.copy(), colormap='viridis',normalize_by=history_level+1) # ,normalize_by=history_level+1


                    # display on leds
                    drawMatrix(self,matrix_to_display)
                    offset_canvas = self.matrix.SwapOnVSync(offset_canvas)
                    self.usleep(10 * 1000)










# Main function
if __name__ == "__main__":
    simple_square = SimpleSquare()
    if (not simple_square.process()):
        simple_square.print_help()
