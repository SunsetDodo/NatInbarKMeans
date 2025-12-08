import sys

EPSILON = 0.001
ITERATIONS = 400
NEGATIVE_START_VALUE = -1


def parse_file(data):
	matrix = []
	temp_row = []
	lined_data = data.split('\n')
	lined_data = lined_data[:-1]
	for row in lined_data: #should chane to any seperator
		temp_vector = row.split(',')
		for scalar in temp_vector:
			temp_row.append(float(scalar))
		matrix.append(temp_row)
		temp_row = []
	return matrix
		

def assign_vectors(matrix, centroids):
	centroids_assignments = {i: [centroids[i],[]] for i in range(k)}
	for vector in matrix:
		centroid_assignment = NEGATIVE_START_VALUE
		min_for_vector = NEGATIVE_START_VALUE
		for centroid_index in range(len(centroids)): 
			temp_distance = distance(vector, centroids[centroid_index])
			if (centroid_assignment < 0) & (min_for_vector < 0):
				centroid_assignment = centroid_index
				min_for_vector = temp_distance
			elif temp_distance < min_for_vector:
				centroid_assignment = centroid_index 
				min_for_vector = temp_distance
		centroids_assignments[centroid_assignment][1].append(vector)
	return centroids_assignments


def recreate_centroids(centroids_assignments_dict):
	for i in range(k):
		new_centroid = calculate_muk(centroids_assignments_dict[i][1])
		centroids_assignments_dict[i][0] = new_centroid


def calculate_muk(list_of_vectors):
	vector_sum = calc_vector_sum(list_of_vectors)
	len_of_list_of_vectors = len(list_of_vectors)
	normalized_vector = normalize_vector(vector_sum, len_of_list_of_vectors)
	return normalized_vector

def calc_vector_sum(list_of_vectors):
	sum_vector = [0 for i in range(k)] 
	for i in range(k):
		sum_for_i = 0
		for vector in list_of_vectors:
			sum_for_i += vector[i]
		sum_vector[i] = sum_for_i
	return sum_vector

def normalize_vector(vector, scalar):
	new_vector = vector.copy()
	for index in range(len(new_vector)):
		new_vector[index] = new_vector[index]/ scalar
	return new_vector

def initialize_centroids(k, matrix):
	return matrix[:k]

def distance(vector1, vector2):
	sum_of_vector = 0
	for i in range(len(vector1)):
		sum_of_vector += (vector1[i] - vector2[i])**2
	return sum_of_vector**(1/2)

def minimal_distance(previous_centroids: dict, new_centroids: dict):
	maximum = NEGATIVE_START_VALUE
	for i in range(k):
		temp_distance = distance(previous_centroids[i], new_centroids[i][0])
		if maximum < temp_distance:
			maximum = temp_distance
	return maximum


def Kmins(k, iters, matrix):
	iterations = 0;
	centroids = initialize_centroids(k, matrix)
	centroids_assignments_dict = assign_vectors(matrix, centroids)
	while(True):
		list_of_previous_centroids = {centroid: centroids_assignments_dict[centroid][0] for centroid in centroids_assignments_dict.keys()}
		recreate_centroids(centroids_assignments_dict)
		if(minimal_distance(list_of_previous_centroids, centroids_assignments_dict) < EPSILON):
			break
		if(iterations > iters):
			break
		iterations += 1
	return [centroids_assignments_dict[centroid_index][0] for centroid_index in centroids_assignments_dict.keys()]


if __name__ == "__main__":
	if len(sys.argv) == 3:
		k = int(sys.argv[1])
		iters =  int(sys.argv[2])
		data = sys.stdin.read()
		matrix = parse_file(data)
		centroid_list = Kmins(k, iters, matrix)
		for centroid in centroid_list:
			print(centroid)
	else:
		print("usage, the input should be in the format: <number> <number> > <file_path>")