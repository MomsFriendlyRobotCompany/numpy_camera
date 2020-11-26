# -*- coding: utf-8 -*
##############################################
# The MIT License (MIT)
# Copyright (c) 2020 Kevin Walchko
# see LICENSE for full details
##############################################
import attr
from collections import namedtuple
import numpy as np
np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)
from slurm import storage
from colorama import Fore
from math import atan, pi

FrameSize = namedtuple("FrameSize", "w h")

class PinholeCamera:
    name = None
    K = None
    P = None
    ready = False
    distortion = None

    def __init__(self, file=None):
        if file and isinstance(file, str):
            data = storage.read(file)
            self.set_params(data)
            self.ready = True

    def __str__(self):
        if not self.ready:
            s = f"{Fore.RED}*** Camera is not initialized yet ***{Fore.RESET}"
            return s

        if self.name:
            name = f"Camera: {self.name}"
        else:
            name = "PinHole Camera"

        s = f"[{name}] ------------\n"
        m = str(self.K).replace('\n', "\n    ")
        s += f"  Camera Matrix(K):\n    {m}\n"
        m = str(self.P).replace('\n', "\n    ")
        s += f"  Projection Matrix(P):\n    {m}\n"
        return s

    # def set(self, w, h, f, c, R, t):
    #     """
    #     (w,h): width, height in pixels
    #     f: (fx,fy) focal length in pixels, focal_length [pixels]
    #     c: (cx,cy) principle point in pixels (image center)[pixels]
    #     R: rotation from world to camera
    #     t: translation from world to camera frame [meters]
    #     distortion: distortion parameters from calibration
    #     """
    #     self.shape = FrameSize(w,h)
    #     Rt = np.hstack((R,t))
    #     fx, fy = f
    #     cx, cy = c
    #     self.K = np.array([
    #         [fx,  0, cx],
    #         [ 0, fy, cy],
    #         [ 0,  0,  1]
    #     ])
    #     self.P = K @ Rt

    def set_params(self, params):
        if "name" in params:
            self.name = params["name"]

        if "f" in params and "c" in params:
            fx, fy = params["f"]
            cx, cy = params["f"]

            self.K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ])

        if "R" in params and "t" in params:
            R = np.array(params["R"]).reshape(3,3)
            t = np.array(params["t"]).reshape(3,1)

            # print(R,t)

            Rt = np.hstack((R,t))

        # if "width" in params and "height" in params:
        #     self.shape = FrameSize(params["width"], params["height"])
        if "size" in params:
            w,h = map(int, params["pi"])
            self.shape = FrameSize(w,h)

        if "pi" in params:
            """
            Assumptions made using Pi Camera v2 specs, YMMV

                f = 3.11 mm
                sensor = 3936x2460 um
            """
            # if params["pi"] != "v2.1":
            #     raise Exception("Wrong PiCamera model:", params["pi"])
            w,h = map(int, params["pi"])
            self.shape = FrameSize(w,h)
            # w,h = self.shape
            fx = 3.11*w/(3936e-3)
            fy = 3.11*h/(2460e-3)
            cx = w/2
            cy = h/2
            self.K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ])

        if "distortion" in params:
            self.distortion = np.array(params["distortion"])

        if "channels" in params:
            self.channels = int(params["channels"])

        self.P = self.K @ Rt

        self.ready = True

    # def world2camera(self, points):
    #     """Transforms n points from world coordinates (X) to camera
    #     coordinates (P).
    #
    #     P[3xn] = [R|t][3x4]*X[4xn]
    #
    #     points: world points X[3xn] or X[4xn] where each point
    #             is (x,y,z) or (x,y,z,1) respectively
    #     return: camera points P[3xn]
    #     """
    #     if points.shape[0] == 3:
    #         n = points.shape[1]
    #         points = np.hstack((points, np.ones(n).reshape((n,1))))
    #     return self.Rt @ points

    def project(self, points):
        """
        Transforms world coordinates (X,Y,Z) to 2D image coordinates (u,v)
        using the projection matrix. This function will make 3D points homogenious
        and return 2D points that are in image space (remove the extra 1).

        points2D[3xn] = P[3x4]*points3D[4xn]
        where:
            P = K*[R|t]
                K = camera matrix [3x3]
                R = rotation matrix [3x3]
                t = translation vector [3x1]
            3D points are (X,Y,Z)
            2D points are (u,v)

        points: camera space(x,y,z)[n, 3], this will change to [4xn]
                homogenious coordinates so the math works correctly
        returns: points in image space(u,v)[n, 2]
        """
        points = points.T
        num_pts = points.shape[1]
        # print(points.shape)

        # Change to homogenous coordinate
        points = np.vstack((points, np.ones((1, num_pts))))
        points = self.P @ points
        points /= points[2, :]
        return points[:2,:].T

    def back_project(self, points):
        """
        not done
        """
        raise NotImplemented()

    def fov(self):
        """
        Returns the FOV as (horizontal, vertical) in degrees
        """
        w,h = self.size
        fx = self.K[0,0]
        fy = self.K[1,1]
        fovx = 2*atan(w/2/fx) * 180/pi
        fovy = 2*atan(h/2/fy) * 180/pi
        return (fovx, fovy)


# class StereoCamera:
#     def __init__(self, cam0, cam1, R, baseline, matcher):
#         """
#         cam0: left camera
#         cam1: right camera
#         R: rotation from left to right camera
#         baseline: translation from left to right camera [meters]
#         matcher: some matcher that calculates disparity between images
#         """
#         self.cam0 = cam0
#         self.cam1 = cam1
#         self.R = R
#         self.baseline = baseline
#         self.matcher = matcher
#
#     def F(self):
#         pass
#
#     def E(self):
#         pass
#
#     def disparity(self, img0, img1):
#         return self.matcher(img0, img1)
#     #
#     # def fov(self):
#     #     return 2*atan(c.npix/2./cam0.f)
