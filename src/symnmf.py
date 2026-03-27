import os, sys, numpy as np, pandas as pd

np.random.seed(1234)
VALID_GOALS = ["symnmf", "sym", "ddg", "norm"]
W = np.array([
        [1.0, 0.9, 0.1, 0.05],
        [0.9, 1.0, 0.15, 0.1],
        [0.1, 0.15, 1.0, 0.85],
        [0.05, 0.1, 0.85, 1.0]
    ])
def sym():
    pass
def symnmf(matrix):
    pass
def ddg():
    pass
def norm():
    pass
def parse_inputs(file1) -> np.ndarray:
    try:
        data = np.loadtxt(file1, delimiter=',')
        return data
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


def initialize_h(W, n, k):
    """
        Randomly initialize H with values from [0, 2 * sqrt(m/k)]
        m is the average of all entries of W.
    """
    m = np.mean(W)
    upper_bound = 2 * np.sqrt(m / k)
    H = np.random.uniform(0, upper_bound, size=(n, k))
    return H


def update_h(W, H, beta=0.5):
    """
    Iteratively update H using the multiplicative update rule.
    """
    # Numerator: W * H
    numerator = np.dot(W, H)
    denominator = np.dot(np.dot(H, H.T), H)
    #todo: do we want the zero error check?
    denominator = np.where(denominator == 0, 1e-9, denominator)


    H_new = H * (1 - beta + beta * (numerator / denominator))

    return H_new


def run_symnmf(W, n, k, max_iter=300, epsilon=1e-4):
    """
    Phase 1.4.3: Update H until convergence or max iterations reached.
    """
    H = initialize_h(W, n, k)

    for i in range(max_iter):
        H_new = update_h(W, H)
        diff = np.linalg.norm(H_new - H, ord='fro') ** 2
        if diff < epsilon:
            return H_new
        H = H_new

    return H

def route(goal, matrix):

    match goal:
        case None:
            return
        case "symnmf":
            pass
        case "sym":
            return sym(matrix)
        case "ddg":
            return ddg(matrix)
        case "norm":
            return norm(matrix)


def main():
    args = sys.argv[1:]
    arg_count = len(args)
    if arg_count not in [2, 3]:
        print("error")
        sys.exit(1)
    try:
        k = int(args[0])
        if k < 0:
            raise ValueError
    except ValueError:
        print("error12")
        sys.exit(1)
    if arg_count == 3:
        goal = args[1]
        file_name = args[2]
        if goal not in VALID_GOALS:
            print(f"error2")
            sys.exit(1)
    else:
        goal = None
        file_name = args[1]

    if not file_name.endswith(".txt"):
        print("error3")
        sys.exit(1)

    if not os.path.exists(file_name):
        print(f"error4")
        sys.exit(1)

    matrix = parse_inputs(sys.argv[3])
    route(goal, matrix)

if __name__ == "__main__":
    main()