
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


const int INITIAL_ROW_CAP = 8;
const char COL_DELIMITER = ',';
const double EPSILON = 0.0001;

typedef struct {
    double** matrix;
    int rows;
    int cols;
} Input;


typedef struct {
    double* center;
    double** assigned_vectors;
    int assigned_vectors_count;
    int vector_length;
} Centroid;


size_t str_len(const char* str) {
    int len = 0;
    while (str[len] != '\0') len++;
    return len;
}


double distance(const double* a, const double* b, const int dim) {
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}


double* str_to_double(const char* str, int* out_col_count) {
    const size_t len = str_len(str);
    double* values = malloc(len * sizeof(double));

    int current_col = 0;
    const char* cursor = str;
    char* pEnd = NULL;

    while (1) {
        double val = strtod(cursor, &pEnd);

        if (cursor == pEnd) {
            break;
        }

        values[current_col++] = val;
        cursor = pEnd;

        if (*cursor == COL_DELIMITER) {
            cursor++;
        }
    }

    *out_col_count = current_col;

    if (current_col > 0) {
        double* exact_values = realloc(values, current_col * sizeof(double));
        if (exact_values) {
            values = exact_values;
        }
    }

    return values;
}

Input parse_input() {
    size_t row_cap = INITIAL_ROW_CAP;
    Input input;
    input.cols = 0;
    input.rows = 0;
    input.matrix = malloc(row_cap * sizeof(double*));

    char* line = NULL;
    size_t buffer_len = 0;

    while (getline(&line, &buffer_len, stdin) != -1) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            break;

        int col_count = 0;
        double* row = str_to_double(line, &col_count);

        if (input.cols == 0) {
            input.cols = col_count;
        }

        input.matrix[input.rows++] = row;

        if (input.rows == row_cap) {
            row_cap *= 2;
            double** new_matrix = realloc(input.matrix, row_cap * sizeof(double*));

            if (!new_matrix) {
                exit(1);
            }

            input.matrix = new_matrix;
        }
    }

    free(line);
    if (input.rows > 0) {
        double** new_matrix = realloc(input.matrix, input.rows * sizeof(double*));
        if (new_matrix) {
            input.matrix = new_matrix;
        }
    }

    if (input.matrix == NULL)
        exit(1);

    return input;
}

double* mean_value(double** vectors, const int vector_count, const int vector_length) {
    double* mean = malloc(vector_length * sizeof(double));
    for (int i = 0; i < vector_length; i++) {
        double sum = 0;
        for (int j = 0; j < vector_count; j++) {
            sum += vectors[j][i];
        }
        mean[i] = sum / vector_count;
    }
    return mean;
}

void free_centroids(const Centroid* centroids, const int centroid_count) {
    if (centroids == NULL)
        return;

    for (int i = 0; i < centroid_count; i++) {
        free(centroids[i].center);
        free(centroids[i].assigned_vectors);
    }
}

double max_iteration_delta(const Centroid* old, const Centroid* new, const int k, const int vector_length) {
    if (old == NULL || new == NULL) {
        return INFINITY;
    }

    double max_delta = 0;
    for (int i = 0; i < k; i++) {
        const double d = distance(old[i].center, new[i].center, vector_length);
        max_delta = max_delta > d ? max_delta : d;
    }
    return max_delta;
}

void assign_vectors_to_centroids(Centroid* centroids, const int k, const Input input) {
    if (centroids == NULL || input.matrix == NULL) {
        return;
    }

    for (int i = 0; i < input.rows; i++) {
        double min_distance = INFINITY;
        Centroid* closest_centroid = NULL;
        for (int j = 0; j < k; j++) {
            const double dist = distance(input.matrix[i], centroids[j].center, input.cols);
            if (dist < min_distance) {
                min_distance = dist;
                closest_centroid = &centroids[j];
            }
        }
        if (closest_centroid == NULL) {
            exit(1);
        }
        if (closest_centroid->assigned_vectors_count == 0) {
            closest_centroid->assigned_vectors = malloc(input.rows * sizeof(double*));
        }
        closest_centroid->assigned_vectors[closest_centroid->assigned_vectors_count++] = input.matrix[i];
    }

    for (int i = 0; i < k; i++) {
        centroids[i].assigned_vectors = realloc(centroids[i].assigned_vectors, centroids[i].assigned_vectors_count * sizeof(double*));
    }
}

Centroid* kmeans(const Input input, const int k, const int max_iterations) {
    Centroid* centroids = malloc(k * sizeof(Centroid));
    for (int i = 0; i < k; i++) {
        centroids[i].center = malloc(input.cols * sizeof(double));
        memcpy(centroids[i].center, input.matrix[i], input.cols * sizeof(double));
        centroids[i].assigned_vectors_count = 0;
    }
    assign_vectors_to_centroids(centroids, k, input);

    for (int i = 0; i < max_iterations; i++) {
        const Centroid* old_centroids = centroids;
        centroids = malloc(k * sizeof(Centroid));
        for (int j = 0; j < k; j++) {
            centroids[j].center = mean_value(old_centroids[j].assigned_vectors, old_centroids[j].assigned_vectors_count, input.cols);
            centroids[j].assigned_vectors_count = 0;
            centroids[j].assigned_vectors = NULL;
        }
        assign_vectors_to_centroids(centroids, k, input);

        if (max_iteration_delta(old_centroids, centroids, k, input.cols) < EPSILON) {
            free_centroids(old_centroids, k);
            return centroids;
        }
        free_centroids(old_centroids, k);
    }
    return centroids;

}


int main(int argc, char** argv) {
    // int k = (int)argv[1];
    // int max_iterations = (int)argv[2];
    const int k = 7;
    const int max_iterations = INFINITY;
    const Input d = parse_input();

    const Centroid* centroids = kmeans(d, k, max_iterations);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d.cols; j++) {
            printf("%lf,", centroids[i].center[j]);
        }
        printf("\n");
    }

    return 0;
}