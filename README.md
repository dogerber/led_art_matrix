# LED Art Matrix

Collection of Python codes (using [rpi-rgb-led-matrix](https://learn.adafruit.com/adafruit-rgb-matrix-plus-real-time-clock-hat-for-raspberry-pi/driving-matrices#step-6-log-into-your-pi-to-install-and-run-software-1745233)), which display animations an an 64 x 64 LED matrix. 



## Showcase
Framerate and colors are poorly represented on these videos. Click to expand:
<details >
  <summary>Comets</summary>
<img src="/vid/dg_planets_1.gif" width="300"/>
<img src="/vid/dg_planets_2.gif" width="300"/>

see `code/dg_planets.py`. Variable modes are available with:
- Planets which don't move
- different interactions between particles (purely attractive, attractive and repulsive if close, attractive but bouncing off each other)
- different boundary conditions

</details>

<details >
  <summary>Gaussian Blur</summary>
<img src="/vid/dg_blur.gif" width="300"/>

</details>

<details >
  <summary>Waves</summary>
<img src="/vid/dg_waves.gif" width="300"/>

</details>


<details >
  <summary>Ising Model</summary>
<img src="/vid/dg_ising_model.gif" width="300"/>

</details>

## Materials
- [64 x 64x LED Matrix Adafruit](https://www.adafruit.com/product/4732)
- [Adafruit RGB Matrix HAT + RTC for Raspberry Pi](https://www.adafruit.com/product/2345)
- Raspberry Pi 4
- Laser cut enclosure (see [/enclosure](/enclosure/))


## How to build
- setup a Raspberry pi (i use header-less mode as described [here](https://www.tomshardware.com/reviews/raspberry-pi-headless-setup-how-to,6028.html))
- install rpi-rgb-led-matrix as described [here](https://learn.adafruit.com/adafruit-rgb-matrix-plus-real-time-clock-hat-for-raspberry-pi/driving-matrices#step-6-log-into-your-pi-to-install-and-run-software-1745233)
- either run `code\dg_master.py` (takes a random other script and runs it for a set amount of time) or 
```sudo python CODE_NAME --led-rows=64 --led-cols=64 --led-gpio-mapping=adafruit-hat-pwm --led-pwm-lsb-nanoseconds 50 --led-slowdown-gpio=4 --led-pwm-bits=11 ```

Intended use is to add `code/dg_master.py` to the startup routine of the Raspberry Pi, such that the LED matrix automatically turn on when the system is powered. 








## Resources
- [Adafruit Tutorial to use the HAT](https://learn.adafruit.com/adafruit-rgb-matrix-plus-real-time-clock-hat-for-raspberry-pi)
- [Flickering troubleshooting](https://github.com/hzeller/rpi-rgb-led-matrix/tree/master#troubleshooting)

