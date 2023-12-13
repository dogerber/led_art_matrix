from gpiozero import LightSensor
from time import sleep

# GPIO.setmode(GPIO.BCM)

sensorPin = 25

sensor = LightSensor(sensorPin, charge_time_limit=0.2)

try:
	while True:
		print(sensor.value)
		sleep(0.5)
except KeyboardInterrupt:
	pass
