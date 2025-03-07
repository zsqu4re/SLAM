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
        pass

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """

        num_particles = X_bar.shape[0]
        # weight normalization
        weights = X_bar[:, 3]
        weights = weights/np.sum(weights)

        cumulative_sum = weights[0]

        X_bar_resampled =  np.zeros_like(X_bar)
        r = np.random.uniform(0, 1.0/num_particles) # Generate initial random number

        # Resmapling
        i = 0
        for m in range(num_particles):
            U = r + m/num_particles
            while cumulative_sum < U:
                i += 1
                cumulative_sum += weights[i]
            X_bar_resampled[m] = X_bar[i]

        # X_bar_resampled[:, 3] = 1.0/num_particles # Reallocate uniform weight
        return X_bar_resampled
    
if __name__ == "__main__":
    X_bar = np.load('X_bar.npy')
    re = Resampling()
    X_bar_resampled = re.low_variance_sampler(X_bar)
    print(X_bar_resampled)
