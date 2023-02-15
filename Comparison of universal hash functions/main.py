import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import murmurhash3_32
import seaborn as sns

# Generate Random a,b,c,d with fixed seed

random.seed(1998)
a = random.randrange(1, 1048573)

random.seed(2022)
b = random.randrange(1, 1048573)

random.seed(2020)
c = random.randrange(1, 1048573)

random.seed(5)
d = random.randrange(1, 1048573)

p = 1048573
seed_murmur = 42

# Counters to keep track of bit change for each of the 10 bit
counter = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

counter_3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

counter_4 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

counter_mur = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

probability_2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

probability_3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

probability_4 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

probability_mur = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def two_universal(x):
    return ((a * x + b) % p) % 1024


def three_universal(x):
    return ((a * x ** 2 + b * x + c) % p) % 1024


def four_universal(x):
    return ((a * x ** 3 + b * x ** 2 + c * x + d) % p) % 1024


def murmur_hash(x):
    return murmurhash3_32(x, seed_murmur) % 1024


def flip_bit_at_k(x, k):
    return x ^ (1 << k)


# Generate 5000 31-bit integers
integers = []
for r in range(5000):
    integers.append(random.getrandbits(31))

# Iterate through 5000 integers to generate result
for integer in integers:
    # 2-universal hash function
    result_original_2 = two_universal(integer)
    # flip the lowest 10 bits of integer
    for input_bit in range(10):
        result_flip_2 = two_universal(flip_bit_at_k(integer, input_bit))
        for output_bit_pos in range(10):
            if ((result_original_2 >> output_bit_pos) & 1) ^ ((result_flip_2 >> output_bit_pos) & 1):
                counter[input_bit][output_bit_pos] = counter[input_bit][output_bit_pos] + 1

    # 3-universal hash function
    result_original_3 = three_universal(integer)
    for input_bit in range(10):
        result_flip_3 = three_universal(flip_bit_at_k(integer, input_bit))
        for output_bit_pos in range(10):
            if ((result_original_3 >> output_bit_pos) & 1) ^ ((result_flip_3 >> output_bit_pos) & 1):
                counter_3[input_bit][output_bit_pos] = counter_3[input_bit][output_bit_pos] + 1

    # 4-universal hash function
    result_original_4 = four_universal(integer)
    for input_bit in range(10):
        result_flip_4 = four_universal(flip_bit_at_k(integer, input_bit))
        for output_bit_pos in range(10):
            if ((result_original_4 >> output_bit_pos) & 1) ^ ((result_flip_4 >> output_bit_pos) & 1):
                counter_4[input_bit][output_bit_pos] = counter_4[input_bit][output_bit_pos] + 1

    # murmur3 hash function
    result_original_mur = murmur_hash(integer)
    for input_bit in range(10):
        result_flip_mur = murmur_hash(flip_bit_at_k(integer, input_bit))
        for output_bit_pos in range(10):
            if ((result_original_mur >> output_bit_pos) & 1) ^ ((result_flip_mur >> output_bit_pos) & 1):
                counter_mur[input_bit][output_bit_pos] = counter_mur[input_bit][output_bit_pos] + 1

# Caluclate the probability of individual output bit
for col in range(10):
    for row in range(10):
        probability_2[9 - row][col] = float(counter[col][row]) / 5000
        probability_3[9 - row][col] = float(counter_3[col][row]) / 5000
        probability_4[9 - row][col] = float(counter_4[col][row]) / 5000
        probability_mur[9 - row][col] = float(counter_mur[col][row]) / 5000

# Draw 2-universal
probability_2_numpy = np.array(probability_2)

sns.heatmap(probability_2_numpy, center=0.5, vmin = 0, vmax = 1)

plt.title("2-universal hash function")
plt.colorbar(label="Probability that output bit will change", orientation="vertical")
plt.show()

# Draw 3-universal
probability_3_numpy = np.array(probability_3)

plt.imshow(probability_3_numpy, cmap='inferno', interpolation='nearest', vmin=0.4)

plt.title("3-universal hash function")
plt.colorbar(label="Probability that output bit will change", orientation="vertical")
plt.show()

# Draw 4-universal
probability_4_numpy = np.array(probability_4)

plt.imshow(probability_4_numpy, cmap='inferno', interpolation='nearest', vmin=0.4)

plt.title("4-universal hash function")
plt.colorbar(label="Probability that output bit will change", orientation="vertical")
plt.show()

# Draw Murmur Hash function
probability_mur_numpy = np.array(probability_mur)

plt.imshow(probability_mur_numpy, cmap='inferno', interpolation='nearest', vmin=0.4)

plt.title("murmur hash function")
plt.colorbar(label="Probability that output bit will change", orientation="vertical")
plt.show()