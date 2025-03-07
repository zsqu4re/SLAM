'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
from scipy.integrate import quad
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from math import pi, cos, sin
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
        # self._z_hit = 1 # most likely; assuming sensor values are close to true values
        # self._z_short = 0.1 # if there were obstacles blocking the sensor
        # self._z_max = 0.1 # max range
        # self._z_rand = 100 # random error

        # tuned _z_ parameters
        self._z_hit = .7 # assume sensor values are close to true values    
        self._z_short = 0.2 # there were obstacles blocking the sensor
        self._z_max = 0.01 # don't trust max range
        self._z_rand = 0.09 # account for error in sensor


        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000
        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.2 # -> change map values to binary
        # Used in sampling angles in ray casting
        self._subsampling = 2
        self.map = occupancy_map
        # self.map = occupancy_map[::-1] # flip map to start from bottom left

    def p_hit(self, z_k, z_k_star, sigma):
        p_hit = norm.pdf(z_k, z_k_star, sigma**2)
        return p_hit
    
    def p_hit_integration(self, z_k, z_k_star, sigma):
        return self.p_hit(z_k, z_k_star, sigma)


    def beam_range_finder_model(self, z_t1_arr, x_t1, num_ray):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        # sensor data - divide by 10
        # rays will contain the ray casted distances for indexed ray
        # rays at 0, pi are included
            # so there is will always be a ray at 0, 180, num_ray - 2 rays

        ray_index = int(180 / (num_ray - 1))
        rays = []
        rays.append(z_t1_arr[0])
        i = ray_index - 1
        while i < 180: 
            rays.append(z_t1_arr[int(i)])
            i += ray_index
        q = 1

        # print(f"particle state belief is : {x_t1}")
        # x_t1 = current particle state belief [x, y, theta] : world frame
        step_size = 10

        true_measure, theta_s = self.ray_casting(x_t1, num_ray, step_size, self.map)

        # print(f"true measure is : {true_measure}")
        # print(f"measured are : {rays}")

        # for each ray calculate the probability
        prob = []
        
        eta = 1
        for i in range(num_ray):
            ## calculating _p_hit
            if 0 <= rays[i] <= self._max_range:
                _p_hit = 1/(np.sqrt(2 * pi * self._sigma_hit ** 2)) * np.exp(-0.5 * ((rays[i] - true_measure[i]) ** 2)/ self._sigma_hit ** 2)
            else:
                _p_hit = 0

            ## calculating _p_short
            if 0 <= rays[i] <= true_measure[i]:
                _p_short = eta * self._lambda_short * np.exp(-1 * self._lambda_short * rays[i])
            else:
                _p_short = 0
            
            ## calculating _p_max
            if rays[i] == self._max_range:
                _p_max = 1
            else:
                _p_max = 0
            
            ## calculating _p_rand
            if 0 <= rays[i] <= self._max_range:
                _p_rand = 1/self._max_range
            else:
                _p_rand = 0
            
            p = self._z_hit * _p_hit + self._z_short * _p_short + self._z_max * _p_max + self._z_rand * _p_rand
            prob.append(p)
        
        p_avg = np.mean(prob)
        prob_zt1 = q * p_avg

        # returning true measure distances and ray casted distances for ray casting visualization
        return prob_zt1, np.array(true_measure), np.array(rays), theta_s

    def ray_casting(self, x_t1, num_ray, step_size, map):
        if num_ray < 3:
            print("Number of rays need to be larger than 3")
            return 
        ### location passed in 8000 x 8000 coordinates

        # conversion to 800 x 800 coordinates
        # http://www.cse.yorku.ca/~amana/research/grid.pdf
        x_exac = x_t1[0] / 10
        y_exac = x_t1[1] / 10

        # accounting for the 25cm offset from the center of the robot
        x_exac = x_exac + 2.5 * cos(x_t1[2])
        y_exac = y_exac + 2.5 * sin(x_t1[2])
        o_exac = [x_exac, y_exac]

        x_coor = int(x_exac) 
        y_coor = int(y_exac)
        o = [x_coor, y_coor] # in 800 x 800 cell coordinates
        theta = x_t1[2] # in radians

        theta_s = []

        # odd number case
        if num_ray % 2 == 1:
            theta_diff = pi / (num_ray - 1)
            theta_s.append(theta)
            i = 1
            while i < num_ray:
                theta_s.append(theta + theta_diff * i)
                i += 1

        else: # even number case (must include -90, 90 degrees)
            theta_diff = pi/(num_ray - 1)
            theta_s.append(theta + pi/2)
            i = 1
            while i < num_ray:
                theta_s.append(theta + pi/2 - theta_diff * i)
                i += 1

        # target position calculation
        e_n = []
        for theta in theta_s:
            # print(theta)
            if theta >= 0 and theta <= pi/2:
                e_n.append((x_coor + self._max_range * cos(theta), y_coor + self._max_range * sin(theta))) 
            elif theta >= pi/2 and theta <= pi:
                e_n.append((x_coor - self._max_range * cos(theta), y_coor + self._max_range * sin(theta)))
            elif theta >= -pi and theta <= -pi/2:
                e_n.append((x_coor - self._max_range * cos(theta), y_coor - self._max_range * sin(theta)))
            else:
                e_n.append((x_coor + self._max_range * cos(theta), y_coor - self._max_range * sin(theta)))

        # unit vectors 
        dir = []
        for idx, e in enumerate(e_n):
            # unit vector = e - o / ||e - o ||
            unit_vector = (e[0] - o_exac[0], e[1] - o_exac[1]) / np.sqrt((e[0]-o_exac[0])**2 + (e[1]-o_exac[1])**2)


            current_location = o_exac
            target_location = e

            obs_dist, obs_loc = self.cell_distance(unit_vector, o_exac, target_location, step_size, map)
            dir.append(obs_dist) # returning raycasted distances    

        # print(f"ray casting distances are: {dir}")
        return dir, theta_s


    def cell_distance(self, unit_vector, current_location, target_location, stepsize, map):
        
        original_location = current_location
        original_cell = int(current_location[0]), int(current_location[1])

        cur_cell = original_cell
        cur_loc = current_location.copy()
        
        target_location = target_location
        target_cell = int(target_location[0]), int(target_location[1])

        while cur_cell[0] != target_cell[0] or cur_cell[1] != target_cell[1]:
            #  prevent leaving map boundaries
            if cur_cell[0] < .5 or cur_cell[1] < .5:
                break
            if cur_cell[0] >= 799.5 or cur_cell[1] >= 799.5:
                break

            # occupied cell
            # index is inversed because map takes in y, x
            if map[cur_cell[1], cur_cell[0]] == 1 or map[cur_cell[1], cur_cell[0]] > 0.2 or map[cur_cell[1], cur_cell[0]] == -1:
                break

            # unoccupied cell 
            else:
                cur_loc[0] += (stepsize/10) * unit_vector[0] #* delta_x_sign
                cur_loc[1] += (stepsize/10) * unit_vector[1] #* delta_y_sign

            # update current cell
            cur_cell = round(cur_loc[0]), round(cur_loc[1])
        distance = np.sqrt((abs(cur_loc[0]) - abs(original_location[0]))**2 + (abs(cur_loc[1]) - abs(original_location[1]))**2)


        return (distance * 10), cur_loc
                


            
if __name__ == "__main__":
    # sample map: 10x10 map
    map = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
    # map = map[::-1] ## inversing map

    sensor_model = SensorModel(map)
    sample_x_t1 = np.array([53, 68, pi])
    num_ray = 4
    step_size = 0.01 # in cm 0 ~ 10

    sensor_model.ray_casting(sample_x_t1, num_ray, step_size, map)


