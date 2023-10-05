import numpy as np
import random
from time import time

class Cluster:
    """Class representing an individual cluster."""

    def __init__(self):
        self.points = []
        self.score = None  # Score for the cluster

    @property
    def size(self):
        return len(self.points)

    @property
    def position(self):
        if not self.points:
            return None
        avg_position = np.mean(self.points, axis=0)
        return tuple(map(int, avg_position))

    def add_point(self, point):
        self.points.append(point)

class DBSCAN:
    """Class implementing the DBSCAN clustering algorithm."""

    def __init__(self, grid, eps=1.5, min_samples=0, alpha=1, beta=1):
        self.grid = grid
        self.eps = eps
        self.min_samples = min_samples
        self.clusters = []
        self.alpha = alpha
        self.beta = beta
        self.best_cluster = None
        self.best_score = float('-inf')

    def fit(self, reference_position):
        labels = np.zeros_like(self.grid)
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if labels[x, y] != 0 or self.grid[x, y] != -1:
                    continue

                neighbors = self.region_query((x, y))

                if len(neighbors) < self.min_samples:
                    labels[x, y] = -1
                else:
                    new_cluster = Cluster()
                    self.expand_cluster_DFS((x, y), neighbors, new_cluster, labels)
                    self.clusters.append(new_cluster)
                    
                    # Calculate the score for the new cluster
                    new_score = self.calculate_score(new_cluster, reference_position)
                    new_cluster.score = new_score  # Save the score in the cluster instance

                    # Update best_cluster if this cluster's score is higher
                    if new_score > self.best_score:
                        self.best_score = new_score
                        self.best_cluster = new_cluster

    def region_query(self, point):
        x, y = point
        neighbors = []

        x_min = max(0, x - int(self.eps))
        x_max = min(self.grid.shape[0], x + int(self.eps) + 1)
        y_min = max(0, y - int(self.eps))
        y_max = min(self.grid.shape[1], y + int(self.eps) + 1)

        for nx in range(x_min, x_max):
            for ny in range(y_min, y_max):
                if (nx, ny) == (x, y):
                    continue
                dist = ((nx - x)**2 + (ny - y)**2)**0.5
                if dist <= self.eps and self.grid[nx, ny] == -1:
                    neighbors.append((nx, ny))

        return neighbors

    def expand_cluster_DFS(self, point, neighbors, cluster, labels):
        """Expand cluster using Depth-First Search."""
        # Depth first search typically seams to be faster and more consistent for scarce undiscovered tiles
        x, y = point
        labels[x, y] = 1
        cluster.add_point(point)
        
        # Recursive DFS implementation
        def dfs(p):
            x, y = p
            for nx, ny in self.region_query(p):
                if labels[nx, ny] == 0:
                    labels[nx, ny] = 1
                    cluster.add_point((nx, ny))
                    dfs((nx, ny))
                elif labels[nx, ny] == -1:
                    labels[nx, ny] = 1
                    cluster.add_point((nx, ny))

        dfs(point)
    
    def expand_cluster_BFS(self, point, neighbors, cluster, labels):
        """Expand cluster using Breadth-First Search."""
        x, y = point
        labels[x, y] = 1
        cluster.add_point(point)
        
        queue = [point]  # Initialize with the starting point
        
        while queue:
            current_point = queue.pop(0)
            for nx, ny in self.region_query(current_point):
                if labels[nx, ny] == 0:
                    labels[nx, ny] = 1
                    cluster.add_point((nx, ny))
                    queue.append((nx, ny))
                elif labels[nx, ny] == -1:
                    labels[nx, ny] = 1
                    cluster.add_point((nx, ny))

    def calculate_score(self, cluster, position):
        """Calculate score of cluster. Score = alpha * size - beta * distance"""
        distance = ((cluster.position[0] - position[0])**2 + (cluster.position[1] - position[1])**2)**0.5
        return self.alpha * cluster.size - self.beta * distance

if __name__ == "__main__":

    #  Testgrid
    n_cols = 20
    n_rows = 20

    fgrid = []
    for row in range(n_rows):
        row = []
        for col in range(n_cols):
            content = random.randint(-1,1)
            if content == -1:
                row.append(content)

            else:
                row.append(0)
        fgrid.append(row)

    grid = np.array(fgrid)

    print(grid)
    print('\n')

    t0 = time()

    dbscan = DBSCAN(grid)
    n_iterations = 4
    for i in range(n_iterations):
        x_pos,y_pos = random.randint(0,n_rows),random.randint(0,n_cols)
        dbscan.clusters = []
        dbscan.fit((x_pos,y_pos))
        print(x_pos,y_pos)

        cluster_info = [{"size": cluster.size, "position": cluster.position, "score": cluster.score} for cluster in dbscan.clusters]
        clusters = {cluster.position: cluster.size for cluster in dbscan.clusters}

        ggrid = []
        for row in range(n_rows):
            grow = []
            for col in range(n_cols):
                if (row,col) in clusters.keys():
                    grow.append(clusters[(row,col)])
                else:
                    grow.append(0)
            ggrid.append(grow)
        
        gggrid = np.array(ggrid)
        print(gggrid)
        print('')

    

    
        

        print(f"Number of clusters: {len(cluster_info)}")
        for info in cluster_info:
            print(f"Cluster Size: {info['size']}, Position: {info['position']}, Score: {info['score']}")
        print(len(dbscan.clusters))
    
    print(f"avg time per dbscan: {(time()-t0)/n_iterations} ")

    ######



    # dbscan.fit((9,6))

    # cluster_info = [{"size": cluster.size, "position": cluster.position, "score": cluster.score} for cluster in dbscan.clusters]
    # clusters = {cluster.position: cluster.size for cluster in dbscan.clusters}

    # ggrid = []
    # for row in range(n_rows):
    #     grow = []
    #     for col in range(n_cols):
    #         if (row,col) in clusters.keys():
    #             grow.append(clusters[(row,col)])
    #         else:
    #             grow.append(0)
    #     ggrid.append(grow)
    
    # gggrid = np.array(ggrid)
    # print(gggrid)

    # print(f"Number of clusters: {len(cluster_info)}")
    # for info in cluster_info:
    #     print(f"Cluster Size: {info['size']}, Position: {info['position']}, Score: {info['score']}")

    # print(len(dbscan.clusters))
