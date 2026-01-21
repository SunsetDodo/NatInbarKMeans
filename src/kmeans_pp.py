import numpy as np
import pandas as pd
import argparse
import mykmeanssp
import sys

def choose_centroids(df, k, seed):
    np.random.seed(seed)

    x = df.to_numpy(dtype=float)
    n = x.shape[0]

    chosen = []
    first = np.random.choice(n)
    chosen.append(first)

    centroids = [x[first]]

    for _ in range(1, k):
        dists = np.full(n, np.inf)

        for c in centroids:
            diff = x - c
            d = np.sqrt(np.sum(diff ** 2, axis=1))
            dists = np.minimum(dists, d)

        probs = dists / dists.sum()
        nxt = np.random.choice(n, p=probs)

        chosen.append(nxt)
        centroids.append(x[nxt])

    keys = df.index.to_numpy()[chosen]
    return np.array(centroids), keys


def parse_inputs(file1, file2) -> pd.DataFrame:
    try:
        np1 = np.loadtxt(file1, delimiter=',')
        np2 = np.loadtxt(file2, delimiter=',')
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

    df1 = pd.DataFrame(np1, columns=["key"] + list(range(np1.shape[1] - 1)))
    df2 = pd.DataFrame(np2, columns=["key"] + list(range(np2.shape[1] - 1)))
    return df1.merge(df2, how="inner", on="key").sort_values("key").set_index("key")

def valid_k(value):
    try:
        return int(value)
    except ValueError:
        print("Incorrect number of clusters!")
        sys.exit(1)

def valid_iter(value):
    try:
        return int(value)
    except ValueError:
        print("Incorrect maximum iteration!")
        sys.exit(1)

def valid_eps(value):
    try:
        return float(value)
    except ValueError:
        print("Incorrect epsilon!")

class ErrorOccuredParser(argparse.ArgumentParser):
    def error(self, message):
        print("An Error Has Occurred")
        sys.exit(1)

def main():
    parser = ErrorOccuredParser()
    parser.add_argument("k", type=valid_k)
    parser.add_argument("iter", type=valid_iter, nargs="?", default=300)
    parser.add_argument("eps", type=valid_eps)
    parser.add_argument("file1", type=str)
    parser.add_argument("file2", type=str)
    parser.add_argument("--seed", type=int, default=1234)

    try:
        args = parser.parse_args()
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

    df = parse_inputs(args.file1, args.file2)

    if not (1 < args.k < df.shape[0]):
        print("Incorrect number of clusters!")
        sys.exit(1)

    if not (1 < args.iter < 800):
        print("Incorrect maximum iteration!")
        sys.exit(1)

    if args.eps < 0:
        print("Incorrect epsilon!")
        sys.exit(1)

    initial_centers, chosen_keys = choose_centroids(df, args.k, args.seed)

    print(",".join(str(int(k)) for k in chosen_keys))

    fitted = mykmeanssp.fit(
        df.to_numpy(dtype=float).tolist(),
        initial_centers.tolist(),
        args.k,
        args.iter,
        args.eps,
    )

    for center in fitted:
        print(",".join(f"{v:.4f}" for v in center))

if __name__ == "__main__":
    main()
