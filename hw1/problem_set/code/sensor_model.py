'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
# from bresenham import bresenham

from map_reader import MapReader
from motion_model import MotionModel
from resampling import Resampling


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map, lookup_table):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 0.9   # 0.7
        self._z_short = 0.11    # 0.2
        self._z_max = 0.05     # 0.01
        self._z_rand = 800   # 0.09

        self._sigma_hit = 100   # 2500
        self._lambda_short = 0.1   # 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 3   # 2

        # The relative distance from robot's EOM to the Lidar (cm)
        self._r_dist = 25

        # Resolution of the each Square of the Map (cm)
        self._map_resolution = 10

        self.map = occupancy_map
        self.lookup_table = lookup_table
    
    def WrapToPi(self, angle):
        angle_wrapped = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
        return angle_wrapped

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        prob_zt1 = 1.0
        z_t = np.array(z_t1_arr[::self._subsampling])
        z_star = self.get_raycast_val(x_t1)
        prob_zt1 = self.calculate_probabilities(z_t, z_star)
        return prob_zt1
    
    def calculate_probabilities(self, z_t, z_star):
        """
        Compute the Weighted Sum of the Four Types of Measurement Error
        param[in] z_t : laser range readings [array of 180/subsampling values] at time t
        param[in] z_star : true map range value in all directions for a given robot pose
        """
        # Initialize the probability
        prob = 1
        for k in range(len(z_t)):
            # Four probabilities
            p_hit = self.prob_hit(z_t[k], z_star[k])
            p_short = self.prob_short(z_t[k], z_star[k])
            p_max = self.prob_max(z_t[k])
            p_rand = self.prob_rand(z_t[k])

            p = self._z_hit*p_hit + self._z_short*p_short + self._z_max*p_max + self._z_rand*p_rand

            prob = prob* p

        return prob
    
    def get_raycast_val(self, x_curr):
        """
        Index into Raycasting Array to get Position
        param[in] x_curr : Current Pose of the Robot w.r.t the World Frame
        param[out] dists : Raycasting Distance for Current Pose
        """
        # Position in world frame
        [x, y, theta] = x_curr

        # Scale to map (pixel) frame
        x1 = (x + self._r_dist * np.cos(theta))// self._map_resolution
        y1 = (y + self._r_dist*np.sin(theta))//self._map_resolution
        x_map, y_map = int(x1), int(y1)

        # Correspond to the closest integer theta (in degress)
        beam_angle = theta - (np.pi/2) + (np.arange(0,180, self._subsampling)*np.pi/180)
        beam_ang_deg = ((beam_angle * (180 / np.pi)) % 360).astype(int)

        # Look up the table
        return self.lookup_table[y_map, x_map, beam_ang_deg]
    
    def prob_hit(self, z_t, z_star):
        """
        Calculate p_hit Gaussian distribution probability
        """
        if 0 <= z_t <= self._max_range:
            return (1/np.sqrt(2*np.pi*(self._sigma_hit**2)))*np.exp((-0.5)*(((z_t-z_star)**2)/(self._sigma_hit**2)))
        return 0

    def prob_short(self, z_t, z_star):
        """
        Calculate the p_short exponential distribution probability
        """
        if 0 <= z_t <= z_star:
            return self._lambda_short * np.exp(-self._lambda_short * z_t)
        return 0

    def prob_max(self, z_t):
        """
        Calculate the p_max probability
        """
        return 1.0 if z_t >= self._max_range else 0.0

    def prob_rand(self, z_t):
        """
        Calculate the p_rand uniform distribution probability
        """
        return 1.0 / self._max_range if 0 <= z_t < self._max_range else 0.0
    
