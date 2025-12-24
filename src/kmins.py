import sys

from dask.dataframe.methods import values
from sympy import false

EPSILON = 0.001
ITERATIONS = 400
NEGATIVE_START_VALUE = -1


class Vector:
    values: list[float] = None
    centroid_uid = None

    def __init__(self, values):
        self.values = values

    def __repr__(self):
        s = ''
        for value in self.values:
            s += str(value) + ", "
        return s
    def copy(self):
        copied_values = self.values.copy()
        return Vector(copied_values)

class Centroid(Vector):
    uid = None

    def __init__(self, values, uid):
        self.values = values
        self.uid = uid
        self.centroid_uid = uid

    def __repr__(self):
        s = f'uid: {self.uid}, values: '
        for value in self.values:
            s += str(value) + ", "
        return s
class CentroidList:
    centroids: list[Centroid] = None
    assigned_vectors: dict[str: list[Vector]] = dict()

    def __init__(self, list_of_centrids):
        self.centroids = list_of_centrids

    def flush(self):
        self.centroids = None
        self.assigned_vectors = dict()

    def set_centroids(self, centroids):
        self.centroids =  centroids

    def add_vector_assignment(self, vector, uid):
        if uid in self.assigned_vectors.keys():
            self.assigned_vectors[uid].append(vector)
        else:
            self.assigned_vectors[uid] = [vector]

    def copy(self):
        centroids_list = self.centroids.copy()
        return CentroidList(centroids_list)

    def __getitem__(self, item):
        return self.centroids[item]


def parse_file(data):
    matrix = []
    temp_row = []
    lined_data = data.split('\n')
    lined_data = lined_data[:-1]
    for row in lined_data: #should chane to any seperator
        temp_vector = row.split(',')
        for scalar in temp_vector:
            temp_row.append(float(scalar))
        v = Vector(temp_row)
        matrix.append(v)
        temp_row = []
    return matrix


def kmeans(k: int, iters: int, matrix: list[Vector]):
    iterations = 0
    list_of_centroids = initialize_centroids(k, matrix)
    assign_vectors(matrix, list_of_centroids)
    while True:
        list_of_prev_centroids = list_of_centroids.copy()
        list_of_centroids = recreate_centroids(list_of_centroids)
        empty_centroids = check_for_empty_centroid(list_of_centroids)
        if len(empty_centroids)!=0:
            initialize_centroids()

        if abs(minimal_distance(list_of_prev_centroids, list_of_centroids)) < EPSILON:
            break
        if iterations >= iters:
            break
        iterations += 1
    return list_of_centroids

def initialize_centroids(k, matrix):
    list_of_centroids = CentroidList([Centroid(matrix[i].values, i) for i in range(k)])
    return list_of_centroids

def assign_vectors(matrix :list[Vector] , centroids: CentroidList):
    for vector in matrix:
        centroid_assignment = NEGATIVE_START_VALUE
        min_for_vector = NEGATIVE_START_VALUE
        for centroid in centroids.centroids:
            temp_distance = distance(vector, centroid)
            if (centroid_assignment < 0) & (min_for_vector < 0):
                centroid_assignment = centroid.uid
                min_for_vector = temp_distance
            elif temp_distance < min_for_vector:
                centroid_assignment = centroid.uid
                min_for_vector = temp_distance
        centroids.add_vector_assignment(vector, centroid_assignment)

def distance(vector1: Vector, vector2: Vector):
    sum_of_vector = 0.0
    for i in range(len(vector1.values)):
        sum_of_vector += (float(vector1.values[i]) - float(vector2.values[i]))**2
    return float(sum_of_vector**(1/2))

def recreate_centroids(list_of_centroids: CentroidList):
    new_list_of_centroids = []
    for i in range(k):
        temp_centroid = calculate_mean_value(list_of_centroids.assigned_vectors[i])
        new_centroid = Centroid(temp_centroid.values, i)
        new_list_of_centroids.append(new_centroid)
    list_of_centroids.flush()
    list_of_centroids.set_centroids(new_list_of_centroids)
    assign_vectors(matrix, list_of_centroids)
    return list_of_centroids

def calculate_mean_value(list_of_vectors):
    vector_sum = calc_vector_sum(list_of_vectors)
    len_of_list_of_vectors = len(list_of_vectors)
    normalized_vector = normalize_vector(vector_sum, len_of_list_of_vectors)
    return normalized_vector

def calc_vector_sum(list_of_vectors):
    vector_len = len(list_of_vectors[0].values) # maybe not that robust
    sum_vector = [0 for i in range(vector_len)]
    for i in range(vector_len):
        sum_for_i = 0
        for vector in list_of_vectors:
            sum_for_i += vector.values[i]
        sum_vector[i] = sum_for_i
    return Vector(sum_vector)

def normalize_vector(vector: Vector, scalar):
    new_vector = vector.copy()
    for index in range(len(new_vector.values)):
        new_vector.values[index] = new_vector.values[index]/ scalar
    return new_vector

def minimal_distance(previous_centroids: CentroidList, new_centroids: CentroidList):
    maximum = NEGATIVE_START_VALUE
    for i in range(k):
        temp_distance = distance(previous_centroids[i], new_centroids[i])
        if maximum < temp_distance:
            maximum = temp_distance
    return maximum

def check_for_empty_centroid(centroid_list: CentroidList): #need to check
    list_of_empty_centroids = set()
    for i in centroid_list.assigned_vectors.keys():
        if not centroid_list.assigned_vectors[i]:
            list_of_empty_centroids.add(str(i))
    return  list_of_empty_centroids

def fix_missing_centroids(cetroid_list: CentroidList, missing_centroids: set[str]): #need to check
    counter = 0
    for i in missing_centroids:
        cetroid_list.centroids[int(i)] = matrix[counter]
        cetroid_list.centroids[int(i)].centroid_uid = i


if __name__ == "__main__":
    k = int(sys.argv[1])
    iters = 400
    if len(sys.argv) == 3:
        iters = int(sys.argv[2])
    data = sys.stdin.read()
    matrix = parse_file(data)
    centroid_list = kmeans(k, iters, matrix)
    for centroid in centroid_list.centroids:
        print(centroid)