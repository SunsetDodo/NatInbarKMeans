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
} Centroid;

size_t str_len(const char* str) {
    size_t len = 0;
    while (str[len] != '\0') len++;
    return len;
}

double distance(const double* a, const double* b, int dim) {
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

double* str_to_double(const char* str, int* out_col_count) {
    size_t len = str_len(str);
    double* values = malloc(len * sizeof(double));
    if (!values) exit(1);

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

Input parse_input(void) {
    size_t row_cap = INITIAL_ROW_CAP;
    Input input;
    input.cols = 0;
    input.rows = 0;
    input.matrix = malloc(row_cap * sizeof(double*));
    if (!input.matrix) exit(1);

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

        if (input.rows == (int)row_cap) {
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

    return input;
}

double* mean_value(double** vectors, int vector_count, int vector_length) {
    // TODO - check vector_count == 0
    double* mean = malloc(vector_length * sizeof(double));
    if (!mean) exit(1);

    for (int i = 0; i < vector_length; i++) {
        double sum = 0;
        for (int j = 0; j < vector_count; j++) {
            sum += vectors[j][i];
        }
        mean[i] = sum / vector_count;
    }
    return mean;
}

void free_input(Input* input) {
    if (!input || !input->matrix) return;
    for (int i = 0; i < input->rows; i++) {
        free(input->matrix[i]);
    }
    free(input->matrix);
    input->matrix = NULL;
    input->rows = 0;
    input->cols = 0;
}

void free_centroids(Centroid* centroids, int centroid_count) {
    if (centroids == NULL) {
        return;
    }

    for (int i = 0; i < centroid_count; i++) {
        free(centroids[i].center);
        free(centroids[i].assigned_vectors);
    }

    free(centroids);
}

double max_iteration_delta(const Centroid* old, const Centroid* newc,
                           const int k, const int vector_length) {
    if (old == NULL || newc == NULL) {
        return INFINITY;
    }

    double max_delta = 0;
    for (int i = 0; i < k; i++) {
        const double d = distance(old[i].center, newc[i].center, vector_length);
        if (d > max_delta) {
            max_delta = d;
        }
    }
    return max_delta;
}

void assign_vectors_to_centroids(Centroid* centroids, int k, const Input input) {
    if (centroids == NULL || input.matrix == NULL) {
        return;
    }

    for (int i = 0; i < k; i++) {
        free(centroids[i].assigned_vectors);
        centroids[i].assigned_vectors = NULL;
        centroids[i].assigned_vectors_count = 0;
    }

    int first_idx = -1;
    // From Forum - it can be assumed that max of 1 centroid is empty at each iteration, so we'll assign
    // all vectors from 1 to n, check for empty centroid and then assign it to it if so else assign it normally.
    for (int i = 0; i < input.rows; i++) {
        double min_distance = INFINITY;
        int best_idx = -1;

        for (int j = 0; j < k; j++) {
            const double dist = distance(input.matrix[i], centroids[j].center, input.cols);
            if (dist < min_distance) {
                min_distance = dist;
                best_idx = j;
            }
        }

        if (best_idx < 0) {
            exit(1);
        }

        if (i == 0) {
            first_idx = best_idx;
            continue;
        }

        Centroid* c = &centroids[best_idx];

        if (c->assigned_vectors_count == 0) {
            c->assigned_vectors = malloc(input.rows * sizeof(double*));
            if (!c->assigned_vectors) exit(1);
        }

        c->assigned_vectors[c->assigned_vectors_count++] = input.matrix[i];
    }

    int empty_idx = -1;
    for (int i = 0; i < k; i++) {
        if (centroids[i].assigned_vectors_count > 0) {
            double** shrunk = realloc(
                centroids[i].assigned_vectors,
                centroids[i].assigned_vectors_count * sizeof(double*)
            );
            if (shrunk) {
                centroids[i].assigned_vectors = shrunk;
            }
        } else {
            empty_idx = i;
        }
    }

    // Add first vector to the proper Centroid
    first_idx = empty_idx == -1 ? first_idx : empty_idx;
    Centroid* c = &centroids[first_idx];
    if (c->assigned_vectors_count == 0) {
        c->assigned_vectors = malloc(input.rows * sizeof(double*));
        if (!c->assigned_vectors) exit(1);
    }
    c->assigned_vectors[c->assigned_vectors_count++] = input.matrix[first_idx];

}

Centroid* kmeans(const Input input, const int k, const int max_iterations) {
    Centroid* centroids = malloc(k * sizeof(Centroid));
    if (!centroids) exit(1);

    for (int i = 0; i < k; i++) {
        centroids[i].center = malloc(input.cols * sizeof(double));
        if (!centroids[i].center) {
            for (int j = 0; j < i; j++) {
                free(centroids[j].center);
                free(centroids[j].assigned_vectors);
            }
            free(centroids);
            return NULL;
        }
        memcpy(centroids[i].center, input.matrix[i], input.cols * sizeof(double));
        centroids[i].assigned_vectors_count = 0;
        centroids[i].assigned_vectors = NULL;
    }

    assign_vectors_to_centroids(centroids, k, input);

    for (int iter = 0; iter < max_iterations; iter++) {
        Centroid* old_centroids = centroids;

        centroids = malloc(k * sizeof(Centroid));
        if (centroids == NULL) {
            free_centroids(old_centroids, k);
            return NULL;
        }

        for (int j = 0; j < k; j++) {
            centroids[j].center = mean_value(
                old_centroids[j].assigned_vectors,
                old_centroids[j].assigned_vectors_count,
                input.cols
            );
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

int main(void) {
    const int k = 7;
    const int max_iterations = 400;
    Input d = parse_input();

    Centroid* centroids = kmeans(d, k, max_iterations);
    if (!centroids) {
        free_input(&d);
        return 1;
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d.cols; j++) {
            printf("%.4f",centroids[i].center[j]);
            if (j + 1 < d.cols) printf(",");
        }
        printf("\n");
    }

    free_centroids(centroids, k);
    free_input(&d);
    return 0;
}
