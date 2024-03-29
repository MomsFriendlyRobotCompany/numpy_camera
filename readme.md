# Numpy Camera

![Python](https://github.com/MomsFriendlyRobotCompany/numpy_camera/workflows/Python/badge.svg)
![GitHub](https://img.shields.io/github/license/MomsFriendlyRobotCompany/numpy_camera)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/numpy_camera)
![PyPI](https://img.shields.io/pypi/v/numpy_camera)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/numpy_camera)
![PyPI - Downloads](https://img.shields.io/pypi/dm/numpy_camera)

**Look away! Still in dev**

Simple threaded camera that doesn't need OpenCV. It is setup to use
`picamera` interface on the Raspberry Pi to grab images. If you are not using
a Raspberry Pi with the PiCamera, then you can't use this library.

## Why?

Getting OpenCV on a Raspberry Pi historically is hard, but now
there is a `pip` library, so it is easier. However, if there 
is some reason why you can't, don't, won't use that, this can
be used.

## Install

```
pip install -U numpy_camera
```

## Usage

```
c = ThreadedCamera((640,480))
c.start()        # starts internal loop
frame = c.read() # numpy array
c.stop()         # stops internal loop
c.join()         # gathers back up the thread
```

# MIT License

**Copyright (c) 2020 Kevin J. Walchko**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
