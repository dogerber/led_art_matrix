#!/usr/bin/env python
from samplebase import SampleBase

import numpy as np
from matplotlib import cm


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

            
    


        # ----------------------- Program -------------------------------------------
        while True:


            matrix_size = 3
            matrix_grayscale = np.random.randint(0,255,size=(matrix_size,matrix_size))*0
            matrix_start = np.zeros((matrix_size,matrix_size,3),dtype=np.uint8)
            matrix_end = matrix_start.copy()

            matrix_start[:, :matrix_size//2, 0] = 255  # Set red channel to maximum

            matrix_end[:,:,2] = 255


            # display on leds
            total_increments = 10
            for i in range(total_increments):
                matrix_i = drawMatrixSmoothTransition(matrix_start,matrix_end,i, total_increments)
                drawMatrix(self,matrix_i)
                offset_canvas = self.matrix.SwapOnVSync(offset_canvas)
                self.usleep(10 * 1000)
                print(matrix_i)


# NOT WORKING



# Main function
if __name__ == "__main__":
    simple_square = SimpleSquare()
    if (not simple_square.process()):
        simple_square.print_help()
