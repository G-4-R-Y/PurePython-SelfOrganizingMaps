import random
random.seed(42)

import math_utils

class SOM():
    def __init__(self, map_width: int, map_height: int, num_features: int, distance: int):
        self.cells      = []
        self.init_cells(map_width, map_height, num_features)
        self.dist       = distance
        self.epochs     = 200 # at least 320 samples to surpass 
                               # the "at least 500 times the 
                               # number of cells as steps" 
                               # threshold, in the paper
        self.neighbours = int(map_width/2) # maybe: 
                                           # [int(map_width/2), 
                                           # int(map_height/2)]

    def init_cells(self, map_width:int, map_height:int, num_features:int):
        for i in range(map_height):
            cell_row = []
            for j in range(map_width):
                cell = Cell(num_features, i+1, j+1)
                cell_row.append(cell)
            self.cells.append(cell_row)
                
    def fit(self, data: list):
        features = data.features
        total_steps = len(features) * self.epochs
        t           = 1
        alpha       = 0.9
        neighbours  = self.neighbours

        for epoch in range(self.epochs):
            print(f"Epoch: {epoch+1}")
            print(f"Step: {t+1}")

            for feature_vector in features:
                self.iter(feature_vector, neighbours, alpha, t)
                t = t + 1
                if t < 1001:
                    # alpha inversely proportional to time step t
                    alpha = 0.1*(1-t/1000)
                    neighbours = int(self.neighbours_decay(t))
                else:
                    alpha = 0.01 * (1 - t / (total_steps - 1000))
                    neighbours = 1

    def iter(self, feature_vector: list, neighbours: int, alpha: float, t: int):
        bmu = self.choose_bmu(feature_vector)
        self.update(bmu, feature_vector, neighbours, alpha, t)

    def neighbours_decay(self, t):
        # linearly decay from neighbours to 1
        # n(t) = -t * (N-1)/1000 + N
        return -1 * t * (self.neighbours - 1) / 1000 \
            + self.neighbours

    def choose_bmu(self, feature_vector: list):
        best_matching_unit = self.cells[0][0]
        min_distance = self.calc_distance(feature_vector, best_matching_unit.weights)
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(self.cells[i]):
                dist = self.calc_distance(feature_vector, cell.weights)
                if dist < min_distance:
                    best_matching_unit = self.cells[i][j]
        return best_matching_unit

    def calc_distance(self, features: list, weights: list):
        if self.dist == 0:
            return math_utils.supreme_dist(features, weights)
        elif self.dist == 1:
            return math_utils.manhattan_dist(features, weights)
        elif self.dist == 2:
            return math_utils.euclidean_dist(features, weights)
        else:
            print("Invalid distance, please select either 0, 1 \
                or 2.\nCall the program with -h flag to obtain \
                more info on parameter options.")
            raise ValueError

    def update(self, bmu: object, feature_vector: list, neighbours: int, alpha: float, t: int):
        bmu_coords       = [bmu.i_coord, bmu.j_coord]

        neighbours_up    = neighbours if bmu_coords[0]-neighbours >= 0 else bmu_coords[0]
        neighbours_down  = neighbours if bmu_coords[0]+neighbours < len(self.cells) \
                            else len(self.cells) - 1 - bmu_coords[1]
        neighbours_right = neighbours if bmu_coords[1]-neighbours >= 0 else bmu_coords[1]
        neighbours_left  = neighbours if bmu_coords[1]+neighbours < len(self.cells[0]) \
                            else len(self.cells[0]) - 1 - bmu_coords[1]

        for i, row in enumerate(self.cells[bmu_coords[0]-neighbours_up:bmu_coords[0]+neighbours_down]):
            for j, neighbour in enumerate(row[bmu_coords[1]-neighbours_right:bmu_coords[1]+neighbours_left]):
                neighbour_coord = [neighbour.i_coord, neighbour.j_coord]
                euler = 2.71
                theta = euler ** (-self.calc_distance(bmu_coords, neighbour_coord)/2 * self.neighbours_decay(t)**2)
                distance = self.calc_distance(feature_vector, neighbour.weights)
                neighbour.update_weights(alpha, distance, theta)
         
    def predict(self, data: list):
        bmus = [self.choose_bmu(feature_vector) for feature_vector in data]
        outputs = [[bmu.i_coord, bmu.j_coord] for bmu in bmus]
        self.show_map(data)
        return outputs

    def show_map(self, data: list):
        maps = []
        for idx, feature_vector in enumerate(data):
            print(f"Map for row {idx}:")
            instance_map = []
            for i, row in enumerate(self.cells):
                dists = []
                for j, cell in enumerate(self.cells[i]):
                    dist = self.calc_distance(feature_vector, cell.weights)
                    dists.append(dist)
                print(dists)
                instance_map.append(dists)
            maps.append(instance_map)
        #return maps


class Cell():
    def __init__(self, num_features: int, i: int, j: int):
        self.weights = []
        self.init_weights(num_features)
        self.i_coord = i
        self.j_coord = j

    def init_weights(self, num_features: int):
        for i in range(num_features):
            self.weights.append(random.gauss(0, 1))

    def update_weights(self, alpha: float, distance: float, theta: float):
        self.weights = [weight + alpha * theta * distance/10000 for weight in self.weights]