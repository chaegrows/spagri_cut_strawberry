import numpy as np
import matplotlib.pyplot as plt

class GridFromLocalStructureWithCorrection:
    def __init__(self, spacing=1.0, tolerance=0.5):
        self.spacing = spacing
        self.tol = tolerance

    def apply_rigid_transform(self, points, angle_deg, translation):
        rad = np.deg2rad(angle_deg)
        R = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad),  np.cos(rad)]
        ])
        return (R @ points.T).T + translation

    def find_neighbors(self, p0, points):
        neighbors = []
        for point in points:
            if np.array_equal(point, p0):
                continue
            dist = np.linalg.norm(point - p0)
            if abs(dist - self.spacing) < self.tol:
                neighbors.append(point)
        return neighbors

    def compute_directions(self, p0, neighbors):
        vectors = [p - p0 for p in neighbors]
        if len(vectors) < 2:
            return None, None
        # 가장 수직에 가까운 두 개를 x, y로 간주
        scores = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                dot = abs(np.dot(vectors[i]/np.linalg.norm(vectors[i]),
                                 vectors[j]/np.linalg.norm(vectors[j])))
                scores.append((abs(dot), vectors[i], vectors[j]))
        scores.sort(key=lambda x: x[0])  # dot product가 작은 = 더 수직
        return scores[0][1] / np.linalg.norm(scores[0][1]), scores[0][2] / np.linalg.norm(scores[0][2])

    def estimate_grid(self, p0, x_vec, y_vec):
        grid = []
        for j in range(6):
            for i in range(6):
                pt = p0 + x_vec * i * self.spacing + y_vec * j * self.spacing
                grid.append((i, j, pt))
        return grid

    def project_to_grid_indices(self, points, p0, x_vec, y_vec):
        grid_indices = {}
        for p in points:
            d = p - p0
            i = int(round(np.dot(d, x_vec) / self.spacing))
            j = int(round(np.dot(d, y_vec) / self.spacing))
            grid_indices[(i, j)] = p
        return grid_indices

    def reconstruct_grid_from_projection(self, grid_indices, p0, x_vec, y_vec):
        i_vals = [i for (i, _) in grid_indices]
        j_vals = [j for (_, j) in grid_indices]
        i_min, i_max = min(i_vals), min(i_vals) + 5
        j_min, j_max = min(j_vals), min(j_vals) + 5

        grid = []
        for j in range(j_min, j_max + 1):
            for i in range(i_min, i_max + 1):
                pt = p0 + i * self.spacing * x_vec + j * self.spacing * y_vec
                grid.append(pt)
        return np.array(grid)

    def visualize(self, observed_points, reconstructed_grid):
        plt.figure(figsize=(6, 6))
        plt.scatter(observed_points[:, 0], observed_points[:, 1], c='red', label='Observed Points')
        grid_pts = np.array([pt for (_, _, pt) in reconstructed_grid])
        plt.scatter(grid_pts[:, 0], grid_pts[:, 1], c='blue', alpha=0.3, label='Estimated Grid')
        plt.legend()
        plt.axis('equal')
        plt.title("Grid with Direction Correction")
        plt.show()

if __name__ == '__main__':
    ref_grid = np.array([[x, y] for y in range(6) for x in range(8)])
    indices = [0, 1, 2, 3, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 44, 45, 46, 47]
    partial = ref_grid[indices]

    model = GridFromLocalStructureWithCorrection(spacing=30.0, tolerance=5.0)
    observed = model.apply_rigid_transform(partial, angle_deg=10, translation=np.array([5.0, 7.0]))

    for p0 in observed:
      neighbors = model.find_neighbors(p0, observed)
      if len(neighbors) < 5:
          continue
      x_vec, y_vec = model.compute_directions(p0, neighbors)

    if x_vec is None or y_vec is None:
        print("x, y 방향 추정을 위한 충분한 이웃이 없습니다.")
        os.exit()

    grid_index_map = model.project_to_grid_indices(observed, p0, x_vec, y_vec)
    x_vec_corr, y_vec_corr = model.correct_directions(p0, rough_grid, observed)

    if x_vec_corr is None or y_vec_corr is None:
        print("보정을 위한 점이 충분하지 않습니다.")
        os.exit()

    corrected_grid = model.estimate_grid(p0, x_vec_corr, y_vec_corr)
    model.visualize(observed, corrected_grid)
