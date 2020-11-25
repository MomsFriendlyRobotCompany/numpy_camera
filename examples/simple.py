#!/usr/bin/env python3
import numpy as np
import numpy_camera as nc

"""
name - str
size - image size, int(w,h)
pi - int(w,h), only need R and t
R - rotation, np.array(1,9)
t - translation, np.array(1,3)
distortion - np.array(1,5)
"""
data = {
    "name": "Left",
    # "pi": "v2.1",
    # "width": 640,
    # "height": 480,
    "pi": [640, 480],
    "channels": 1,
    "R": [1,0,0, 0,1,0, 0,0,1],
    "t": [0,0,0],
    "distortion": [1,2,3,4,5]
}


cam = nc.pinhole_camera.PinholeCamera()
cam.set_params(data)
print(cam)

# project points into camera image
# points in 3D space [n, 3] -> [[x,y,z], [x,y,z] ...]
pts = np.array([
    [0,0,3],
    [.3,.3,4]
])
uv = cam.project(pts)
print(uv)
