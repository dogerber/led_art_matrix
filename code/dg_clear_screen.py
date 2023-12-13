import numpy as np
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import time

# LED matrix parameters
matrix_width = 64
matrix_height = 64


# LED matrix setup
options = RGBMatrixOptions()
options.rows = matrix_height
options.cols = matrix_width
options.chain_length = 1
options.parallel = 1
options.pwm_bits = 11 # default 11, minimum 1, lower should be less flickering
options.pwm_lsb_nanoseconds = 50
# options.limit_refresh_rate_hz = 60
options.gpio_slowdown = 4
options.hardware_mapping = 'adafruit-hat-pwm'  # or 'adafruit-hat'

matrix = RGBMatrix(options=options)




def drawMatrix(matrix,matrix_to_display):
    for i in range(matrix_to_display.shape[0]):
        for j in range(matrix_to_display.shape[1]):
            matrix.SetPixel(i,j,matrix_to_display[i,j,0],matrix_to_display[i,j,1],matrix_to_display[i,j,2])



# Loop to continuously update LED colors based on Voronoi regions
try:
        matrix_to_display = np.zeros((matrix_height,matrix_width,3))

        # display on leds
        drawMatrix(matrix,matrix_to_display)
        time.sleep(1)
        print("screen clearing done.")

except KeyboardInterrupt:
    matrix.Clear()
