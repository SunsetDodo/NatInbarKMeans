import argparse
import sys

from typing import List, Tuple

EPSILON = 0.001
ITERATIONS = 400
NEGATIVE_START_VALUE = -1


class Vector:
    values: list[float] = None

    def __init__(self, values):
        self.values = values

    def __repr__(self):
        return ','.join(f"{value:.4f}" for value in self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __add__(self, other):
        return Vector([self[i] + other[i] for i in range(len(self))])

    def __sub__(self, other):
        return Vector([self[i] - other[i] for i in range(len(self))])

    def copy(self):
        return Vector(self.values.copy())

class Centroid:
    center: Vector = None
    vectors: List[Vector] = None

    def __init__(self, center: Vector, vectors: List[Vector] = None):
        self.center = center

        if vectors is None:
            self.vectors = []
        else:
            self.vectors = vectors

    def __getitem__(self, item):
        return self.vectors[item]

    def copy(self):
        return Centroid(self.center.copy(), [vector.copy() for vector in self.vectors])

    def append(self, vector: Vector):
        self.vectors.append(vector)


def parse_file(data: str):
    matrix = []
    for row in data.split('\n')[:-1]:
        matrix.append(Vector([float(val) for val in row.split(',')]))
    return matrix


def kmeans(k: int, iters: int, matrix: list[Vector]):
    centroids = initialize_centroids(k, matrix)
    assign_vectors(centroids, matrix)

    for i in range(iters):
        prev_centroids = centroids
        centroids = recreate_centroids(centroids)
        if abs(minimal_distance(prev_centroids, centroids)) < EPSILON:
            break

        assign_vectors(centroids, matrix)
        del prev_centroids

    return centroids


def initialize_centroids(k, matrix):
    return [Centroid(matrix[i]) for i in range(k)]


def closest_centroid(vector: Vector, centroids: List[Centroid]) -> Centroid:
    if not centroids:
        raise RuntimeError("No centroids provided.")

    min_distance = distance(vector, centroids[0].center)
    closest = centroids[0]

    for centroid in centroids[1:]:
        distance_to_centroid = distance(vector, centroid.center)
        if distance_to_centroid < min_distance:
            min_distance = distance_to_centroid
            closest = centroid

    return closest


def assign_vectors(centroids: List[Centroid], matrix: list[Vector]):
    first_centroid_assignment = None
    for i, vector in enumerate(matrix):
        to_assign = closest_centroid(vector, centroids)

        if i == 0:
            first_centroid_assignment = to_assign
            continue

        to_assign.append(vector)

    empty_centroid = check_for_empty_centroid(centroids)
    if empty_centroid:
        empty_centroid.append(matrix[0])
    else:
        first_centroid_assignment.append(matrix[0])


def distance(vector1: Vector, vector2: Vector):
    sum_of_vector = 0.0
    for i in range(len(vector1)):
        sum_of_vector += (vector1.values[i] - vector2.values[i]) ** 2
    return sum_of_vector ** 0.5


def recreate_centroids(centroids: List[Centroid]) -> List[Centroid]:
    new_centroids = []
    for centroid in centroids:
        center = mean_value(centroid.vectors)
        new_centroids.append(Centroid(center))

    return new_centroids

def mean_value(vectors: list[Vector]):
    vector_len = len(vectors[0])
    sum_vector = [0 for _ in range(vector_len)]
    for i in range(vector_len):
        sum_vector[i] = sum(vectors[j][i] / len(vectors) for j in range(len(vectors)))
    return Vector(sum_vector)


def minimal_distance(previous_centroids: List[Centroid], new_centroids: List[Centroid]):
    maximum = NEGATIVE_START_VALUE

    for i, prev in enumerate(previous_centroids):
        new = new_centroids[i]
        dist = distance(prev.center, new.center)
        if maximum < dist:
            maximum = dist

    return maximum

def check_for_empty_centroid(centroids: List[Centroid]) -> Centroid or None:
    for c in centroids:
        if len(c.vectors) == 0:
            return c

    return None


def main():
    if len(sys.argv) not in [2, 3]:
        print("An Error Has Occurred")
        sys.exit(1)

    k_raw = sys.argv[1]
    if not k_raw.replace('.', '').isnumeric() or k_raw.count('.') > 1 or k_raw.endswith('.') or int(float(k_raw)) != float(k_raw):
        print("Incorrect number of clusters!")
        sys.exit(1)

    k = int(float(k_raw))

    max_iters = 400
    if len(sys.argv) == 3:
        max_iters_raw = sys.argv[2]
        if not max_iters_raw.replace('.', '').isnumeric() or max_iters_raw.count('.') > 1 or max_iters_raw.endswith('.') or int(float(max_iters_raw)) != float(max_iters_raw):
            print("Incorrect maximum iteration!")
            sys.exit(1)

        max_iters = int(float(max_iters_raw))

    if not (1 < max_iters < 800):
        print("Incorrect maximum iteration!")
        sys.exit(1)

    data = sys.stdin.read()
    matrix = parse_file(data)

    if not (1 < k < len(matrix)):
        print("Incorrect number of clusters!")
        sys.exit(1)

    centroids = kmeans(k, max_iters, matrix)
    for centroid in centroids:
        print(centroid.center)

if __name__ == "__main__":
    main()