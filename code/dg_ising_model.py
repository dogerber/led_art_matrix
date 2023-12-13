#!/usr/bin/env python
from samplebase import SampleBase
import numpy as np
from matplotlib import cm
from scipy.ndimage import convolve


class IsingModel:
    def __init__(self, shape, temperature=1.0):
        self.shape = shape
        self.temperature = temperature
        self.spins = np.random.choice([-1, 1], shape)
        self.energy_matrix = np.zeros(shape=shape)

    def energy(self, i, j, multiplier=1):
        spin = self.spins[i, j]*multiplier
        neighbors = [
            self.spins[(i - 1) % self.shape[0], j],
            self.spins[(i + 1) % self.shape[0], j],
            self.spins[i, (j - 1) % self.shape[1]],
            self.spins[i, (j + 1) % self.shape[1]]
        ]
        return -spin * np.sum(neighbors)

    def metropolis(self, i, j):
        current_energy = self.energy(i, j)
        new_spin = -self.spins[i, j]
        new_energy = self.energy(i, j,-1)
        delta_energy = new_energy - current_energy

        self.energy_matrix[i,j] = current_energy

        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / self.temperature):
            self.spins[i, j] = new_spin
            

    def simulate(self, steps):
        for _ in range(steps):
            i, j = np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1])
            self.metropolis(i, j)




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
                ising_model = IsingModel(matrix_shape, temperature=0.1)

                for _ in range(10000):
                    ising_model.simulate(100)

                    # ramp temperature up
                    if True:
                        ising_model.temperature += 0.005

                    if True: #blur
                        matrix_to_display = gaussian_blur(ising_model.spins.copy(),sigma=2, kernel_size=3)
                    else:
                        matrix_to_display = ising_model.spins.copy()


                    if True: # color according to temperature
                        matrix_to_display*=ising_model.temperature


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
