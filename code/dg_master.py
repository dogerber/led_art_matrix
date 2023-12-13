import subprocess
import time
import random

# Specify the maximum allowed running time in seconds
max_run_time = 5*60  # [seconds] 5 minutes

# Specify the script you want to run and its arguments
script_paths = [
    "dg_ising_model.py",
    "dg_particles.py",
    "dg_voronoi.py",
    # Add more script paths as needed
]

while True: 
    # Randomly select a script from the list
    selected_script = random.choice(script_paths)

    arguments = [
        "--led-rows=64",
        "--led-cols=64",
        "--led-gpio-mapping=adafruit-hat-pwm",
        "--led-pwm-lsb-nanoseconds=50",
        "--led-slowdown-gpio=2",
        "--led-pwm-bits=8"
    ]
    # Create a list with the command to run, including the script and its arguments
    command = ["sudo", "python", selected_script] + arguments

    try:
        # Use subprocess to run the command with a timeout
        process = subprocess.run(command, capture_output=True, text=True, timeout=max_run_time)

        # Check the output and return code
        print("Output:", process.stdout)
        print("Error:", process.stderr)
        print("Return Code:", process.returncode)

    except subprocess.TimeoutExpired:
        print(f"Process timed out after {max_run_time} seconds.")
        process = subprocess.run(["sudo", "python", "dg_clear_screen.py"], capture_output=True,text=True)
