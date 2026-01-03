import numpy as np, pandas as pd, sys


def choose_centroids(df, k, seed=None):
    rng = np.random.default_rng(seed)

    # Step 1: Choose the first center uniformly at random
    centroids = [df.sample(n=1, random_state=seed).values[0]]

    for _ in range(1, k):
        distances = []
        for _, row in df.iterrows():
            point = row.values
            d_sq = min(np.sum((point - c) ** 2) for c in centroids)
            distances.append(d_sq)

        distances = np.array(distances)

        probs = distances / distances.sum()
        next_centroid_idx = rng.choice(len(df), p=probs)

        centroids.append(df.iloc[next_centroid_idx].values)

    return np.array(centroids)

def parse_file(file1, file2):
    np1 = np.loadtxt(file1, delimiter=',')
    np2 = np.loadtxt(file2, delimiter=',')
    len_np1 = len(np1[0])
    len_np2 = len(np2[0])
    df1 = pd.DataFrame(np1, columns=["key"] + [i for i in range(len_np1-1)])
    df2 = pd.DataFrame(np2, columns=["key"] + [i for i in range(len_np2-1)])
    df = df1.merge(df2, how="inner",on='key').sort_values('key').set_index('key')
    return df





# def main():
#     if len(sys.argv) not in [5,6]:
#         print("An Error Has Occurred")
#         sys.exit(1)
#
#     k_raw = sys.argv[1]
#     if not k_raw.replace('.', '').isnumeric() or k_raw.count('.') > 1 or k_raw.endswith('.') or int(float(k_raw)) != float(k_raw):
#         print("Incorrect number of clusters!")
#         sys.exit(1)
#
#     k = int(float(k_raw))
#
#     max_iters = 400
#     if len(sys.argv) == 5:
#         max_iters_raw = sys.argv[2]
#         if not max_iters_raw.replace('.', '').isnumeric() or max_iters_raw.count('.') > 1 or max_iters_raw.endswith('.') or int(float(max_iters_raw)) != float(max_iters_raw):
#             print("Incorrect maximum iteration!")
#             sys.exit(1)
#
#         max_iters = int(float(max_iters_raw))
#         file1 = sys.argv[4]
#         file2 = sys.argv[5]
#     else:
#         file1 = sys.argv[3]
#         file2 = sys.argv[4]
#
#
#
#     if not (1 < max_iters < 800):
#         print("Incorrect maximum iteration!")
#         sys.exit(1)
#
#
#     parse_file(file1, file2)
#     #
#     # if not (1 < k < len(matrix)):
#     #     print("Incorrect number of clusters!")
#     #     sys.exit(1)
#     #
#     # centroids = kmeans(k, max_iters, matrix)
#     # for centroid in centroids:
#     #     print(centroid.center)


def main():
    k = 5
    file1 = r"C:\Users\inbar\OneDrive\Documents\degree\Project_software\HW1\NatInbarKMeans\src\tests_ex2\input_1_db_1.txt"
    file2 = r"C:\Users\inbar\OneDrive\Documents\degree\Project_software\HW1\NatInbarKMeans\src\tests_ex2\input_1_db_2.txt"
    df = parse_file(file1, file2)
    print(df)
    a = choose_centroids(df, k)
    print(a)


if __name__ == "__main__":
    main()