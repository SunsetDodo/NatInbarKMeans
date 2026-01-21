#define PY_SSIZE_T_CLEAN

#include <math.h>
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

const int INITIAL_ROW_CAP = 8;
const char COL_DELIMITER = ',';

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

void* mem_chain_realloc(MemNode* node, const size_t new_size) {
    void* new_data;

    if (!node) return NULL;

    new_data = realloc(node->data, new_size);
    if (!new_data && new_size != 0) {
        return NULL;
    }

    node->data = new_data;
    return new_data;
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


#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn))
#endif
void free_and_exit(MemChain* chain, const int code) {
    if (code) {
        printf("An Error Has Occurred");
    }
    mem_chain_free(chain);
    exit(code);
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

static int py_list_to_matrix(
    MemChain* chain,
    PyObject* obj,
    double*** out_mat,
    int* out_rows,
    int* out_cols
) {
    int rows, cols;
    int i, j;
    MemNode* mat_node;
    MemNode* row_nodes_node;
    double** mat;
    MemNode** row_nodes;

    if (!PyList_Check(obj)) return 0;

    rows = (int)PyList_Size(obj);
    if (rows <= 0) return 0;

    PyObject* first_row = PyList_GetItem(obj, 0);
    if (!PyList_Check(first_row)) return 0;

    cols = (int)PyList_Size(first_row);
    if (cols <= 0) return 0;

    mat_node = mem_chain_malloc(chain, rows * sizeof(double*));
    if (!mat_node) return 0;
    mat = (double**)mat_node->data;

    row_nodes_node = mem_chain_malloc(chain, rows * sizeof(MemNode*));
    if (!row_nodes_node) return 0;
    row_nodes = (MemNode**)row_nodes_node->data;

    for (i = 0; i < rows; i++) {
        PyObject* row;
        double* vec;
        row = PyList_GetItem(obj, i);
        if (!PyList_Check(row) || (int)PyList_Size(row) != cols) return 0;

        MemNode* vec_node = mem_chain_malloc(chain, cols * sizeof(double));
        if (!vec_node) return 0;

        vec = (double*)vec_node->data;

        for (j = 0; j < cols; j++) {
            PyObject* item;
            double v;
            item = PyList_GetItem(row, j);
            v = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) return 0;
            vec[j] = v;
        }

        row_nodes[i] = vec_node;
        mat[i] = vec;
    }

    *out_mat = mat;
    *out_rows = rows;
    *out_cols = cols;
    return 1;
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

void assign_vectors_to_centroids(MemChain* chain, Centroid* centroids, const int k, const Input input, int rows, int cols) {
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

    for (i = 0; i < rows; i++) {
        double min_distance;
        int best_idx;

        min_distance = INFINITY;
        best_idx = -1;

        for (j = 0; j < k; j++) {
            double dist;

            dist = distance(input.matrix[i], centroids[j].center, cols);
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
            c->assigned_vectors_node = mem_chain_malloc(chain, rows * sizeof(double*));
            if (!c->assigned_vectors_node) free_and_exit(chain, 1);
            c->assigned_vectors = c->assigned_vectors_node->data;
        }

        c->assigned_vectors[c->assigned_vectors_count++] = input.matrix[i];
    }

    empty_idx = -1;

    for (i = 0; i < k; i++) {
        if (centroids[i].assigned_vectors_count > 0) {
            centroids[i].assigned_vectors = mem_chain_realloc(centroids[i].assigned_vectors_node, centroids[i].assigned_vectors_count * sizeof(double*));
            if (!centroids[i].assigned_vectors) free_and_exit(chain, 1);
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
    } else {
        c->assigned_vectors = mem_chain_realloc(c->assigned_vectors_node, (c->assigned_vectors_count + 1) * sizeof(double*));
        if (!c->assigned_vectors) free_and_exit(chain, 1);
    }
    c->assigned_vectors[c->assigned_vectors_count++] = input.matrix[0];
}

MemNode* kmeans(MemChain* chain, const Input input, int rows, int cols, double** initial_centroids, const int k, const int max_iterations, const double epsilon) {
    int i, j;
    MemNode* centroids_node;
    Centroid* centroids;

    centroids_node = mem_chain_malloc(chain, k * sizeof(Centroid));
    if (!centroids_node) free_and_exit(chain, 1);
    centroids = centroids_node->data;

    for (i = 0; i < k; i++) {
        MemNode* center_node;

        center_node = mem_chain_malloc(chain, cols * sizeof(double));
        if (!center_node) free_and_exit(chain, 1);

        centroids[i].center = center_node->data;
        centroids[i].center_node = center_node;

        for (j = 0; j < cols; j++) {
            centroids[i].center[j] = initial_centroids[i][j];
        }

        centroids[i].assigned_vectors_count = 0;
        centroids[i].assigned_vectors = NULL;
        centroids[i].assigned_vectors_node = NULL;
    }

    assign_vectors_to_centroids(chain, centroids, k, input, rows, cols);

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

            mean_node = mean_value(chain, old_centroids[j].assigned_vectors, old_centroids[j].assigned_vectors_count, cols);

            new_centroids[j].center_node = mean_node;
            new_centroids[j].center = (double*)mean_node->data;

            new_centroids[j].assigned_vectors_node = NULL;
            new_centroids[j].assigned_vectors = NULL;
            new_centroids[j].assigned_vectors_count = 0;
        }

        assign_vectors_to_centroids(chain, new_centroids, k, input, rows, cols);

        if (max_iteration_delta(old_centroids, new_centroids, k, cols) < epsilon) {
            free_centroids(chain, centroids_node, k);
            return new_centroids_node;
        }

        free_centroids(chain, centroids_node, k);
        centroids_node = new_centroids_node;
    }

    return centroids_node;
}

static PyObject* fit(PyObject* self, PyObject* args) {
    PyObject *points_obj, *centroids_obj;
    int k, max_iterations;
    double epsilon;

    MemChain* chain = NULL;
    Input d = {
        .matrix = NULL,
        .vector_nodes = NULL,
        .matrix_head = NULL,
    };
    double** points = NULL;
    double** init_centroids = NULL;
    int n_rows, dim_points;
    int k_rows, dim_centroids;

    MemNode* centroids_node;
    Centroid* centroids;

    if (!PyArg_ParseTuple(args, "OOiid", &points_obj, &centroids_obj, &k, &max_iterations, &epsilon)) {
        return NULL;
    }

    chain = mem_chain_init();
    if (!chain) {
        PyErr_SetString(PyExc_RuntimeError, "Allocation failed");
        return NULL;
    }

    if (!py_list_to_matrix(chain, points_obj, &points, &n_rows, &dim_points)) {
        mem_chain_free(chain);
        PyErr_SetString(PyExc_ValueError, "Invalid points");
        return NULL;
    }

    if (!py_list_to_matrix(chain, centroids_obj, &init_centroids, &k_rows, &dim_centroids)) {
        mem_chain_free(chain);
        PyErr_SetString(PyExc_ValueError, "Invalid centroids");
        return NULL;
    }

    if (k_rows != k) {
        mem_chain_free(chain);
        PyErr_SetString(PyExc_ValueError, "k does not match number of centroids");
        return NULL;
    }

    if (dim_points != dim_centroids) {
        mem_chain_free(chain);
        PyErr_SetString(PyExc_ValueError, "Points and centroids dimension mismatch");
        return NULL;
    }

    if (!(1 < max_iterations && max_iterations < 800)) {
        mem_chain_free(chain);
        PyErr_SetString(PyExc_ValueError, "Incorrect maximum iteration");
        return NULL;
    }

    if (!(1 < k && k < n_rows)) {
        mem_chain_free(chain);
        PyErr_SetString(PyExc_ValueError, "Incorrect number of clusters");
        return NULL;
    }

    d.matrix = points;
    d.vector_nodes = NULL;
    d.matrix_head = NULL;

    centroids_node = kmeans(chain, d, n_rows, dim_points, init_centroids, k, max_iterations, epsilon);
    centroids = (Centroid*)centroids_node->data;

    PyObject* out = PyList_New(k);
    if (!out) {
        free_and_exit(chain, 1);
    }

    for (int i = 0; i < k; i++) {
        PyObject* row = PyList_New(dim_points);
        if (!row) {
            Py_DECREF(out);
            mem_chain_free(chain);
            return NULL;
        }
        for (int j = 0; j < dim_points; j++) {
            PyObject* val = PyFloat_FromDouble(centroids[i].center[j]);
            if (!val) {
                Py_DECREF(row);
                Py_DECREF(out);
                mem_chain_free(chain);
                return NULL;
            }
            PyList_SetItem(row, j, val);
        }
        PyList_SetItem(out, i, row);
    }

    mem_chain_free(chain);
    return out;
}


static PyMethodDef fitMethods[] = {
    {"fit",
        fit,
        METH_VARARGS,
        PyDoc_STR("fit(vectors: list[list[float]], centroids: list[list[float]], k: int, max_iterations: int, epsilon: float) -> list[list[float]]"
                  "\n\nRuns k-means algorithm with a given initial centroids.")
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    fitMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&moduledef);
}
