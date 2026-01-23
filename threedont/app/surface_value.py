import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

__all__ = ['SurfaceValueCalculator']

MINIMUM_SAMPLED_POINTS = 300
N_NEIGHBORS = 3  # 5 neighbors + 1 (the point itself)

class SurfaceValueCalculator:
    def __init__(self, accuracy = 0.01, points=None, resolution=None):
        """
        :param accuracy: the ratio between the total number of points and the randomly selected points to consider in the average distance computation
        :param points: points on which to compute the resolution
        :param resolution: already computed resolution
        """

        self.accuracy = accuracy
        if resolution is not None:
            self.resolution = resolution
        elif points is not None:
            self.resolution = self.compute_resolution(points)
        else:
            raise ValueError("Either points or resolution must be provided.")

    def compute_resolution(self, points) -> float:
        n_points = points.shape[0]
        n_sampled_points = int(n_points * self.accuracy)
        if n_sampled_points < MINIMUM_SAMPLED_POINTS:
            n_sampled_points = min(MINIMUM_SAMPLED_POINTS, n_points) # nse sa mai. il signor Tranquillo è morto becco
        random_indices = [random.randint(0, (points.shape[0] - 1)) for _ in range(n_sampled_points)]
        selected_points = points[random_indices] # Randomly select points

        nn = NearestNeighbors(n_neighbors=N_NEIGHBORS)
        nn.fit(points)
        distances, _ = nn.kneighbors(selected_points)
        distances = distances[:, 2:]  # Remove the first column (self-distances)
        average_distance = np.mean(distances)

        return average_distance

    def compute_surface_value(self, number_of_points) -> float:
        sqrt_3 = 1.73205
        return (sqrt_3 / 4) * self.resolution**2 * number_of_points