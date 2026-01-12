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