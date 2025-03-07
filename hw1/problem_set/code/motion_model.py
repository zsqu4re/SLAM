'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math
import argparse, os
from map_reader import MapReader
from matplotlib import pyplot as plt


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.00005
        self._alpha2 = 0.00005
        self._alpha3 = 0.001
        self._alpha4 = 0.001


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """

        x0, y0, theta0 = u_t0      # odom of t-1
        x1, y1, theta1 = u_t1      # odom of t

        delta_rot1 = np.arctan2(y1 - y0, x1 - x0) - theta0
        delta_trans = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        delta_rot2 = theta1 - theta0 - delta_rot1

        delta_rot1_hat = delta_rot1 - np.random.normal(0, self._alpha1 * (delta_rot1**2) + self._alpha2 * (delta_trans**2))
        delta_trans_hat = delta_trans - np.random.normal(0, self._alpha3 * (delta_trans**2) + self._alpha4 * (delta_rot1**2) + self._alpha4 * (delta_rot2**2))
        delta_rot2_hat = delta_rot2 - np.random.normal(0, self._alpha1 * (delta_rot2**2) + self._alpha2 * (delta_trans**2))

        x, y, theta = x_t0         # particle state of t

        x_prime = x + delta_trans_hat * np.cos(theta + delta_rot1_hat)
        y_prime = y + delta_trans_hat * np.sin(theta + delta_rot1_hat)
        theta_prime = theta + delta_rot1_hat + delta_rot2_hat
        # Wrapping angle to (-pi to pi)
        theta_prime = (theta_prime + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_prime, y_prime, theta_prime])
    
