'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        self._num_particles = 1000
        self._resampling_method = 'low_variance' 


    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        weights = X_bar[:, 3]  # Extract weights
        weights /= np.sum(weights)  # Normalize
        indices = np.random.choice(self._num_particles, size=self._num_particles, p=weights)
        
        # Select new particles based on resampled indices
        X_bar_resampled = X_bar[indices].copy()
        X_bar_resampled[:, 3] = 1.0 / self._num_particles
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """

        num_particles = X_bar.shape[0]
        X_bar_resampled = np.zeros_like(X_bar)

        weights = X_bar[:, 3]  # Extract weights
        weights /= np.sum(weights)  # Normalize weights to sum to 1

        r = np.random.uniform(0, 1 / num_particles)
        c = weights[0]
        i = 0

        for m in range(num_particles):
            U = r + m / num_particles
            while U > c:
                i += 1
                if i >= num_particles:  # Prevent out-of-bounds index
                    i = num_particles - 1
                c += weights[i]
            X_bar_resampled[m] = X_bar[i]

        return X_bar_resampled

