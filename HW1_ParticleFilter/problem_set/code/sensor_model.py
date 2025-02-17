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

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self.occupancy_map = occupancy_map
        self.map_resolution = 10
        self._z_hit = 0.8
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.2

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        Compute the likelihood of observed measurements given the particle's state.
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        x, y, theta = x_t1 
        prob_zt1 = 1.0  

        step_size = 5  
        max_steps = int(self._max_range / step_size)

        for i in range(len(z_t1_arr)): 
            z_k = z_t1_arr[i]  
            angle = theta + np.deg2rad(i - 90)  

           
            z_star_k = self._max_range  
            for step in range(max_steps):
                x_curr = x + step * step_size * np.cos(angle)
                y_curr = y + step * step_size * np.sin(angle)

                # Check if the current position is occupied
                map_x, map_y = int(x_curr / self.map_resolution), int(y_curr / self.map_resolution)
                if 0 <= map_x < self.occupancy_map.shape[1] and 0 <= map_y < self.occupancy_map.shape[0]:
                    if self.occupancy_map[map_y, map_x] > self._min_probability:
                        z_star_k = step * step_size 
                        break
                else:
                    break  

            # Compute probability components
            p_hit = norm.pdf(z_k, loc=z_star_k, scale=self._sigma_hit) if z_k <= self._max_range else 0
            p_short = self._lambda_short * np.exp(-self._lambda_short * z_k) if z_k <= z_star_k else 0
            p_max = 1.0 if z_k >= self._max_range else 0
            p_rand = 1.0 / self._max_range if z_k < self._max_range else 0

            # Compute weighted sum
            p = (self._z_hit * p_hit) + (self._z_short * p_short) + (self._z_max * p_max) + (self._z_rand * p_rand)

            prob_zt1 *= p  # Multiply probabilities (independent beams)

        return prob_zt1  # Final likelihood

