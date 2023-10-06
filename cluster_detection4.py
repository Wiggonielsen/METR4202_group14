import numpy as np
import random

class Cluster:
    def __init__(self):
        self.points = []
        self.score = None

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
    def __init__(self, grid, eps=1, min_samples=4, alpha=1, beta=1):
        
        self.eps = eps
        self.min_samples = min_samples
        self.alpha = alpha
        self.beta = beta

        # Reset cluster data for new iteration
        self.reset_grid(grid)

    def reset_grid(self,grid):
        self.grid = grid

        # reset knowledge of best cluster
        self.best_cluster = None
        self.best_score = float('-inf')

        # Reset clusters
        self.clusters = []


    def fit(self, reference_position):
        labels = np.zeros_like(self.grid)
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if labels[x, y] != 0 or self.grid[x, y] != -1:
                    continue

                neighbors = self.region_query((x, y))

                # Check if this point has any neighboring 0 tiles
                if any(self.grid[nx, ny] == 0 for nx, ny in neighbors):
                    new_cluster = Cluster()
                    self.expand_cluster((x, y), new_cluster, labels)
                    self.clusters.append(new_cluster)
                    
                    new_score = self.calculate_score(new_cluster, reference_position)
                    new_cluster.score = new_score

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
                if dist <= self.eps:
                    neighbors.append((nx, ny))

        return neighbors

    def expand_cluster(self, point, cluster, labels):
        """Expand cluster using Breadth-First Search."""
        x, y = point
        labels[x, y] = 1
        cluster.add_point(point)

        queue = [point]

        while queue:
            current_point = queue.pop(0)
            for nx, ny in self.region_query(current_point):
                if self.grid[nx, ny] == -1 and labels[nx, ny] == 0:
                    labels[nx, ny] = 1
                    cluster.add_point((nx, ny))
                    queue.append((nx, ny))

    def calculate_score(self, cluster, position):
        distance = ((cluster.position[0] - position[0])**2 + (cluster.position[1] - position[1])**2)**0.5
        return self.alpha * cluster.size - self.beta * distance

if __name__ == "__main__":

    #  Testgrid
    n_cols = 22
    n_rows = 22

    fgrid = []

    for row in range(n_rows):
        current_row = []
        for col in range(n_cols):
            if row in list(range(int(n_rows/2-5), int(n_rows/2+5))) and col in list(range(int(n_cols/2-5), int(n_cols/2+5))):
                current_row.append(0)
            else:
                content = random.randint(-1,1)
                if content == -1:
                    current_row.append(content)

                else:
                    current_row.append(2)
        fgrid.append(current_row)

    grid = np.array(fgrid)

    print(grid)
    print('\n')

    dbscan = DBSCAN(grid)
    n_iterations = 3
    for i in range(n_iterations):

        # Create new grid
        fgrid = []

        for row in range(n_rows):
            current_row = []
            for col in range(n_cols):
                if row in list(range(int(n_rows/2-3), int(n_rows/2+3))) and col in list(range(int(n_cols/2-3), int(n_cols/2+3))):
                    current_row.append(0)
                else:
                    content = random.randint(-1,1)
                    if content == -1:
                        current_row.append(content)

                    else:
                        current_row.append(2)
            fgrid.append(current_row)

        grid = np.array(fgrid)

        print(grid)
        print('\n')

        dbscan.reset_grid(grid)

        x_pos,y_pos = random.randint(0,n_rows),random.randint(0,n_cols)
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

        # Can retreve best cluster
        print(dbscan.best_cluster.position)