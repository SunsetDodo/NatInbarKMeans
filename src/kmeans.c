#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

const int INITIAL_ROW_CAP = 8;
const char COL_DELIMITER = ',';
const double EPSILON = 0.0001;

typedef struct MemNode{
    void* data;
    struct MemNode* next;
    struct MemNode* prev;
} MemNode;

typedef struct MemChain{
    MemNode* head;
    MemNode* tail;
} MemChain;

typedef struct {
    double** matrix;
    MemNode** vector_nodes;
    MemNode* matrix_head;

    int rows;
    int cols;
} Input;

typedef struct {
    MemNode* center_node;
    double* center;

    MemNode* assigned_vectors_node;
    double** assigned_vectors;

    int assigned_vectors_count;
} Centroid;


MemChain* mem_chain_init(void) {
    MemChain* chain;

    chain = malloc(sizeof(MemChain));
    chain->head = NULL;
    chain->tail = NULL;
    return chain;
}


MemNode* mem_chain_malloc(MemChain* chain, const size_t size) {
    MemNode* node;
    void* data;

    if (!chain) return NULL;

    node = malloc(sizeof(MemNode));
    if (!node) {
        return NULL;
    }

    data = malloc(size);
    if (!data) {
        free(node);
        return NULL;
    }

    node->data = data;
    node->next = NULL;
    node->prev = chain->tail;

    if (chain->head == NULL) {
        chain->head = node;
        chain->tail = node;
    } else {
        chain->tail->next = node;
        chain->tail = node;
    }

    return node;
}


void* mem_chain_realloc(MemNode* node, size_t new_size) {
    void* new_data;

    if (!node) return NULL;

    new_data = realloc(node->data, new_size);
    if (!new_data) {
        return node->data;
    }

    node->data = new_data;
    return new_data;
}


void mem_chain_free(MemChain* chain) {
    MemNode* curr;

    if (!chain) return;

    curr = chain->head;
    while (curr) {
        MemNode* next;

        next = curr->next;
        free(curr->data);
        free(curr);
        curr = next;
    }

    free(chain);
}

void unlink_node(MemChain* chain, MemNode* node) {
    if (!chain || !node) return;

    if (node->prev) node->prev->next = node->next;
    else chain->head = node->next;

    if (node->next) node->next->prev = node->prev;
    else chain->tail = node->prev;

    free(node->data);
    free(node);
}

void unlink(MemChain* chain, void* ptr) {
    MemNode* curr;

    if (!chain || !ptr) return;

    curr = chain->head;
    while (curr) {
        if (curr->data == ptr) {
            unlink_node(chain, curr);
            return;
        }
        curr = curr->next;
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn))
#endif
void free_and_exit(MemChain* chain, int code) {
    mem_chain_free(chain);
    exit(code);
}


int parse_positive_int(const char* s) {
    char* end;
    long v;

    if (s == NULL || *s == '\0') {
        exit(1);
    }

    v = strtol(s, &end, 10);

    if (*end != '\0' || v <= 0) {
        exit(1);
    }

    return (int)v;
}


void parse_args_positional(int argc, char** argv, int* k, int* max_iterations) {
    int max_iter = 400;

    if (!(argc == 3 || argc == 2)) {
        fprintf(stderr, "Usage: %s <k> [<max_iterations>]\n", argv[0]);
        exit(1);
    }

    if (argc == 3) {
        max_iter = parse_positive_int(argv[2]);
    }

    *k = parse_positive_int(argv[1]);
    *max_iterations = max_iter;
}

size_t str_len(const char* str) {
    size_t len = 0;
    while (str[len] != '\0') len++;
    return len;
}


double distance(const double* a, const double* b, int dim) {
    double sum = 0;
    int i;

    for (i = 0; i < dim; i++) {
        double d;

        d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}


MemNode* str_to_vector_node(MemChain* chain, const char* str, int* out_col_count) {
    size_t len;
    MemNode* values_node;
    double* values;
    int current_col;
    const char* cursor;
    char* pEnd;

    len = str_len(str);

    values_node = mem_chain_malloc(chain, len * sizeof(double));
    if (!values_node) free_and_exit(chain, 1);
    values = values_node->data;

    current_col = 0;
    cursor = str;
    pEnd = NULL;

    while (1) {
        double val;

        val = strtod(cursor, &pEnd);
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
        mem_chain_realloc(values_node, current_col * sizeof(double));
    }

    return values_node;
}

Input parse_input(MemChain* chain) {
    size_t row_cap = INITIAL_ROW_CAP;
    Input input;
    char* line;
    int col_count;
    size_t buffer_len;
    MemNode* matrix_node;
    MemNode* vector_nodes_node;


    input.cols = 0;
    input.rows = 0;

    matrix_node = mem_chain_malloc(chain, row_cap * sizeof(double*));
    if (!matrix_node) free_and_exit(chain, 1);
    input.matrix = matrix_node->data;
    input.matrix_head = matrix_node;

    vector_nodes_node = mem_chain_malloc(chain, row_cap * sizeof(MemNode*));
    if (!vector_nodes_node) free_and_exit(chain, 1);
    input.vector_nodes = vector_nodes_node->data;

    line = NULL;
    buffer_len = 0;

    while (getline(&line, &buffer_len, stdin) != -1) {
        MemNode* row_node;

        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            break;

        col_count = 0;
        row_node = str_to_vector_node(chain, line, &col_count);
        if (input.cols == 0) {
            input.cols = col_count;
        }

        input.vector_nodes[input.rows] = row_node;
        input.matrix[input.rows++] = row_node->data;

        if (input.rows == (int)row_cap) {
            row_cap *= 2;

            input.matrix = mem_chain_realloc(matrix_node, row_cap * sizeof(double*));
            input.vector_nodes = mem_chain_realloc(vector_nodes_node, row_cap * sizeof(MemNode*));

            if (input.matrix == NULL || input.vector_nodes == NULL) {
                free(line);
                free_and_exit(chain, 1);
            }
        }
    }
    free(line);

    if (input.rows > 0) {
        input.matrix = mem_chain_realloc(matrix_node, input.rows * sizeof(double*));
        input.vector_nodes = mem_chain_realloc(vector_nodes_node, input.rows * sizeof(MemNode*));
    }

    return input;
}

MemNode* mean_value(MemChain* chain, double** vectors, int vector_count, int vector_length) {
    MemNode* node;
    double* mean;
    int i, j;

    node = mem_chain_malloc(chain, vector_length * sizeof(double));
    if (!node) free_and_exit(chain, 1);
    mean = node->data;

    for (i = 0; i < vector_length; i++) {
        double sum; sum = 0;
        for (j = 0; j < vector_count; j++) {
            sum += vectors[j][i];
        }
        mean[i] = sum / vector_count;
    }
    return node;
}

void free_centroids(MemChain* chain, MemNode* centroids_node, int centroid_count) {
    int i;
    Centroid* centroids;

    centroids = centroids_node->data;

    if (centroids == NULL) {
        return;
    }

    for (i = 0; i < centroid_count; i++) {
        unlink_node(chain, centroids[i].center_node);
        unlink_node(chain, centroids[i].assigned_vectors_node);
        centroids[i].assigned_vectors_node = NULL;
        centroids[i].assigned_vectors = NULL;
        centroids[i].assigned_vectors_count = 0;
        centroids[i].center_node = NULL;
        centroids[i].center = NULL;
    }

    unlink_node(chain, centroids_node);
}

double max_iteration_delta(const Centroid* old, const Centroid* newc,
                           const int k, const int vector_length) {
    double max_delta;
    int i;

    max_delta = 0;

    if (old == NULL || newc == NULL) {
        return INFINITY;
    }

    for (i = 0; i < k; i++) {
        double d;

        d = distance(old[i].center, newc[i].center, vector_length);
        if (d > max_delta) {
            max_delta = d;
        }
    }
    return max_delta;
}

void assign_vectors_to_centroids(MemChain* chain, Centroid* centroids, int k, const Input input) {
    int i, j;
    int first_idx = -1;
    int empty_idx;
    Centroid* c;

    if (centroids == NULL || input.matrix == NULL) {
        return;
    }

    for (i = 0; i < k; i++) {
        unlink_node(chain, centroids[i].assigned_vectors_node);
        centroids[i].assigned_vectors_node = NULL;
        centroids[i].assigned_vectors = NULL;
        centroids[i].assigned_vectors_count = 0;
    }

    for (i = 0; i < input.rows; i++) {
        double min_distance;
        int best_idx;

        min_distance = INFINITY;
        best_idx = -1;

        for (j = 0; j < k; j++) {
            double dist;

            dist = distance(input.matrix[i], centroids[j].center, input.cols);
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

        c = &centroids[best_idx];

        if (c->assigned_vectors_count == 0) {
            c->assigned_vectors_node = mem_chain_malloc(chain, input.rows * sizeof(double*));
            if (!c->assigned_vectors_node) free_and_exit(chain, 1);
            c->assigned_vectors = c->assigned_vectors_node->data;
        }

        c->assigned_vectors[c->assigned_vectors_count++] = input.matrix[i];
    }

    empty_idx = -1;

    for (i = 0; i < k; i++) {
        if (centroids[i].assigned_vectors_count > 0) {
            centroids[i].assigned_vectors = mem_chain_realloc(centroids[i].assigned_vectors_node, centroids[i].assigned_vectors_count * sizeof(double*));
        } else {
            empty_idx = i;
        }
    }

    first_idx = empty_idx == -1 ? first_idx : empty_idx;
    c = &centroids[first_idx];
    if (c->assigned_vectors_count == 0) {
        c->assigned_vectors_node = mem_chain_malloc(chain, sizeof(double*));
        if (!c->assigned_vectors_node) free_and_exit(chain, 1);
        c->assigned_vectors = c->assigned_vectors_node->data;
    }
    c->assigned_vectors[c->assigned_vectors_count++] = input.matrix[0];
}

MemNode* kmeans(MemChain* chain, const Input input, const int k, const int max_iterations) {
    int i, j;
    MemNode* centroids_node;
    Centroid* centroids;

    centroids_node = mem_chain_malloc(chain, k * sizeof(Centroid));
    if (!centroids_node) free_and_exit(chain, 1);
    centroids = centroids_node->data;

    for (i = 0; i < k; i++) {
        MemNode* center_node;

        center_node = mem_chain_malloc(chain, input.cols * sizeof(double));
        if (!center_node) free_and_exit(chain, 1);

        centroids[i].center = center_node->data;
        centroids[i].center_node = center_node;

        memcpy(centroids[i].center, input.matrix[i], input.cols * sizeof(double));

        centroids[i].assigned_vectors_count = 0;
        centroids[i].assigned_vectors = NULL;
        centroids[i].assigned_vectors_node = NULL;
    }

    assign_vectors_to_centroids(chain, centroids, k, input);

    for (i = 0; i < max_iterations; i++) {
        MemNode* new_centroids_node;
        Centroid* new_centroids;
        Centroid* old_centroids;

        old_centroids = centroids_node->data;

        new_centroids_node = mem_chain_malloc(chain, k * sizeof(Centroid));
        if (!new_centroids_node) free_and_exit(chain, 1);
        new_centroids = new_centroids_node->data;

        for (j = 0; j < k; j++) {
            MemNode* mean_node;

            mean_node = mean_value(chain, old_centroids[j].assigned_vectors, old_centroids[j].assigned_vectors_count, input.cols);

            new_centroids[j].center_node = mean_node;
            new_centroids[j].center_node = mean_node;
            new_centroids[j].center = (double*)mean_node->data;

            new_centroids[j].assigned_vectors_node = NULL;
            new_centroids[j].assigned_vectors = NULL;
            new_centroids[j].assigned_vectors_count = 0;
        }

        assign_vectors_to_centroids(chain, new_centroids, k, input);

        if (max_iteration_delta(old_centroids, new_centroids, k, input.cols) < EPSILON) {
            free_centroids(chain, centroids_node, k);
            return new_centroids_node;
        }

        free_centroids(chain, centroids_node, k);
        centroids_node = new_centroids_node;
        centroids = new_centroids;
    }

    return centroids_node;
}

int main(int argc, char** argv) {
    int k, max_iterations;
    int i, j;
    Input d;
    MemNode* centroids_node;
    MemChain* chain;
    Centroid* centroids;

    parse_args_positional(argc, argv, &k, &max_iterations);

    chain = mem_chain_init();

    d = parse_input(chain);

    centroids_node = kmeans(chain, d, k, max_iterations);
    centroids = centroids_node->data;

    if (!centroids_node) free_and_exit(chain, 1);

    for (i = 0; i < k; i++) {
        for (j = 0; j < d.cols; j++) {
            printf("%.4f", centroids[i].center[j]);
            if (j + 1 < d.cols) printf(",");
        }
        printf("\n");
    }

    free_and_exit(chain, 0);
}
