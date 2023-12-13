#!/usr/bin/env python
do_display_on_matrix = False # display on LED matrix or output as gif in current folder

from samplebase import SampleBase

import numpy as np

import random

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable



if do_display_on_matrix:
    def create_gif(frames, output_filename, target_resolution=(800, 800), duration=100, loop=0):
        # Create a new list to store modified frames
        modified_frames = []

        for frame in frames:
            # Upscale the frame to the target resolution
            upscaled_frame = Image.fromarray(frame).resize(target_resolution, resample=Image.NEAREST)

            # Create a black grid on the upscaled frame
            if False:
                draw = ImageDraw.Draw(upscaled_frame)
                grid_color = (0, 0, 0)  # Black
                grid_size = 2  # Adjust the grid size as needed

                # Draw horizontal grid lines
                for y in range(0, target_resolution[1], grid_size):
                    draw.line([(0, y), (target_resolution[0], y)], fill=grid_color, width=1)

                # Draw vertical grid lines
                for x in range(0, target_resolution[0], grid_size):
                    draw.line([(x, 0), (x, target_resolution[1])], fill=grid_color, width=1)

            # Append the modified frame to the list
            modified_frames.append(upscaled_frame)

        # Save the modified frames as a GIF
        modified_frames[0].save(
            output_filename,
            save_all=True,
            append_images=modified_frames[1:],
            duration=duration,
            loop=loop
        )


class Particle:
    def __init__(self, x, y, speed_x, speed_y, attributes=None):
        self.x = x
        self.y = y
        self.speed = np.array([speed_x, speed_y])
        self.attributes = attributes or {}
        self.memory = [(x, y)]
        self.not_movable = False
        self.color = self.generate_random_color()

    def __str__(self):
        return (f"Particle at ({self.x}, {self.y}), "
                f"Speed: ({self.speed[0]}, {self.speed[1]}), "
                f"Attributes: {self.attributes}, "
                f"Color: {self.color}, "
                f"Not Movable: {self.not_movable}\n")

    def generate_random_color(self):
        return np.random.uniform(0,255,size=(3,))

    def move(self, dt):
        if self.not_movable:
            self.speed = [0.0,0.0]
        self.x += self.speed[0] * dt
        self.y += self.speed[1] * dt
        self.memory.append((self.x, self.y))
        
        # Friction term
        if False:
            # friction_factor = (1- sum(self.speed)/1000)
            # self.speed = self.speed*friction_factor
            friction_force = (-1)*(self.speed[0]**2+self.speed[1]**2)*0.0001 # v^2 
            self.speed += (friction_force)

    def interact(self, other_particle):
        distance = np.sqrt((self.x - other_particle.x)**2 + (self.y - other_particle.y)**2)

        # Attraction force between particles
        force_strength = 0.1*other_particle.attributes.get('mass',0)
        force_direction = np.array([other_particle.x - self.x, other_particle.y - self.y])

        if False: # purely attractive
            force = force_strength * force_direction / distance**2
        elif False: # bounce off
            distance_change = 1
            if distance>=distance_change: # attract
                force = force_strength * force_direction / distance**2 
            else: # bounce
                force = 0
                other_particle.speed = (-1)*other_particle.speed
        else: # attractive, but repulsive when close
            distance_change = 2
            if distance>=distance_change:
                force = force_strength * force_direction / distance**2 
            else:
                 force = (-1)*force_strength * force_direction / distance**2



        # Update particle speed based on the attraction force
        self.speed += (force)/self.attributes.get('mass',0)

        # change color when colliding
        if False:
            if distance<2:
                self.color = other_particle.color




# Functions

def create_random_particles(A):
    particles = []
    for _ in range(A):
        x = random.uniform(0, 63)  # Adjust the range based on your requirements
        y = random.uniform(0, 63)
        speed_x = random.uniform(-1, 1)*1.5
        speed_y = random.uniform(-1, 1)*1.5
        particle = Particle(x, y, speed_x, speed_y, {'mass': random.uniform(1, 10)})  # Random mass for each particle
        particles.append(particle)
    return particles

def heavy_particle(xi,yi, mass_i = 50, color_i = [255,255,255]):
    x = xi 
    y = yi
    speed_x = 0.0
    speed_y = 0.0
    particle = Particle(x, y, speed_x, speed_y, {'mass': mass_i})  # Random mass for each particle
    particle.color = color_i
    particle.not_movable = True
    return particle

def draw_planet(xi,yi,r, mass_i = 50,color_i=[255,255,255],):
    particles = []
    for i in range(0,64):
        for j in range(0,64):
            if (np.sqrt((i-xi)**2+(j-yi)**2)<=r):
                particles.append(heavy_particle(i,j,mass_i = mass_i, color_i = color_i))
    return particles



def particle_step(particles, steps):
    # plt.figure(figsize=(8, 8))
    
    for _ in range(steps):
        for particle in particles:
            # keep in boundary boy
            if True:
                if particle.x < 1:
                    particle.speed = np.array([-particle.speed[0],particle.speed[1]])
                if particle.x > 63:
                    particle.speed = np.array([-particle.speed[0],particle.speed[1]])
                if particle.y < 1:
                    particle.speed = np.array([particle.speed[0],-particle.speed[1]])
                if particle.y > 63:
                    particle.speed = np.array([particle.speed[0],-particle.speed[1]])
            particle.move(0.1)
            for other_particle in particles:
                if particle != other_particle:
                    particle.interact(other_particle)




def rgb_to_hsl(rgb):
    r, g, b = rgb
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    # Calculate hue
    if delta == 0:
        h = 0
    elif cmax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60 * (((b - r) / delta) + 2)
    elif cmax == b:
        h = 60 * (((r - g) / delta) + 4)

    # Calculate lightness
    l = (cmax + cmin) / 2

    # Calculate saturation
    s = 0 if delta == 0 else delta / (1 - abs(2 * l - 1))

    return round(h), round(s * 100), round(l * 100)


def hsl_to_rgb(hsl):
    h, s, l = hsl
    h, s, l = h % 360, min(100, max(0, s)), min(100, max(0, l))
    s /= 100
    l /= 100

    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    elif 300 <= h < 360:
        r, g, b = c, 0, x

    r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255

    return round(r), round(g), round(b)


def map_integer_to_color(value, colormap='viridis', vmin=None, vmax=None):
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=colormap, norm=norm)
    return sm.to_rgba(value)




class SimpleSquare(SampleBase):
    def __init__(self, *args, **kwargs):
        super(SimpleSquare, self).__init__(*args, **kwargs)


    def run(self):
        offset_canvas = self.matrix.CreateFrameCanvas()

        # Arbitrary length A and number of steps
        A = 5 # number of particles
        steps = 10000 # number of time steps
        max_trace = 30 # trace that is plotted

        # Create an array (list) of random particles
        particles_array = create_random_particles(A)


        # add a heavy particle in the middle
        if False:
            x = 31 
            y = 31
            speed_x = 0.0
            speed_y = 0.0
            particle = Particle(x, y, speed_x, speed_y, {'mass': 100})  # Random mass for each particle
            particle.color = [255,255,255]
            particles_array.append(heavy_particle(31,31))
            particles_array.append(heavy_particle(32,32))
            particles_array.append(heavy_particle(32,31))
            particles_array.append(heavy_particle(31,32))


        particles_array += draw_planet(31,43,2.5)
        particles_array += draw_planet(10,5,1)
        particles_array += draw_planet(50,23,1)

        # Output particles to start with
        if False:
            print(*particles_array)

        frames = []

        # let system evolve
        for _ in range(steps):
            # Plot particles trajectories with attraction and color fading for multiple steps
            particle_step(particles_array, 1)

            # clear screen
            offset_canvas.Fill(0,0,0)

            # visualize particles (and traces)
            for particle in particles_array:
                x, y = zip(*particle.memory)
                # change order
                x = x[::-1]
                y = y[::-1]

                mem_trace = min(len(x),max_trace)
                
                for i in reversed(range(1, mem_trace)):
                    dim_factor = (mem_trace-i+1)/mem_trace
                    if i > 4:
                        color_hsl = rgb_to_hsl(particle.color)
                        color_hsl = list(color_hsl)
                        color_hsl[2] = color_hsl[2]*dim_factor

                        # hue
                        # color_hsl[0] = color_hsl[0]+i*5 # rainbow!
                        # color_hsl[0] = sum(particle.speed)
                        # rgb_tail_speed = map_integer_to_color(sum(abs(particle.speed)), colormap='inferno', vmin=0, vmax=10) # depending on speed
                        # color_hsl_tail_speed = rgb_to_hsl(rgb_tail_speed[0:3])
                        # color_hsl[0] = color_hsl_tail_speed[0]

                        rgb_i = hsl_to_rgb(tuple(color_hsl))
                    else: # head part
                        rgb_i = particle.color

                    offset_canvas.SetPixel(int(x[i]),int(y[i]),
                                        rgb_i[0],rgb_i[1],rgb_i[2])
                
            # display 
            if do_display_on_matrix:
                offset_canvas = self.matrix.SwapOnVSync(offset_canvas)
                self.usleep(10 * 1000)
            else:
                frames.append(rgb_matrix)




# Main function
if __name__ == "__main__":
    simple_square = SimpleSquare()
    if (not simple_square.process()):
        simple_square.print_help()