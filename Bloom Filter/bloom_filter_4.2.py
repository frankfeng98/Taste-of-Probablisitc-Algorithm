import sys
from matplotlib import pyplot as plt
from sklearn.utils import murmurhash3_32
from bitarray import bitarray
import math
import numpy as np
import random
import csv
import pandas as pd
import string


class BloomFilter():
    def __init__(self, n, r):
        # Size of the array (round up to the nearest power of 2
        self.size = r

        # Initialize the array
        self.bit_array = bitarray(self.size)

        # Calculate the number of hash functions
        self.hash_number = math.ceil(0.7 * r / n)

        # Initialize all bits to 0
        self.bit_array.setall(0)

    def insert(self, key):
        for i in range(self.hash_number):
            pos = murmurhash3_32(key, i) % self.size
            self.bit_array[pos] = 1

    def test(self, key):
        for i in range(self.hash_number):
            pos = murmurhash3_32(key, i) % self.size
            if self.bit_array[pos] == 0:
                return False
        return True


# Test the bloom filter with url list
data = pd.read_csv('user-ct-test-collection-01.txt', sep="\t")
urllist = data.ClickURL.dropna().unique()

random.seed(40)
testSet_url = random.sample(sorted(urllist), 1000)

random.seed(50)
for randomString in range(1000):
    testSet_url.append(''.join(random.choice(string.ascii_letters) for i in range(13)))

fpRate_list = []
r_size_list = []
memory_usage_list = []

for r_factor in range(15):
    falseCount = 0
    array_size = 377871 * (r_factor + 1)
    testCase = BloomFilter(377871, array_size)
    for url in urllist:
        testCase.insert(url)

    for test_url_pos in range(2000):
        if test_url_pos >= 1000:
            if testCase.test(testSet_url[test_url_pos]):
                falseCount = falseCount + 1

    fpRate = falseCount / 1000
    print(fpRate)
    fpRate_list.append(fpRate)
    r_size_list.append(array_size)
    memory_usage_list.append(sys.getsizeof(testCase.bit_array))

with open("result_memory.txt", "w") as txt_file:
    txt_file.write("FP rate: " + "\n")
    for line in fpRate_list:
        txt_file.write(" ".join(str(line)) + ", ")
    txt_file.write("\n")
    txt_file.write("Memory Used: " + "\n")
    for memory in memory_usage_list:
        txt_file.write(" ".join(str(memory)) + ", ")

# Begin to plot the correlation graph
y = pd.Series(fpRate_list)
x = pd.Series(r_size_list)
correlation = y.corr(x)

plt.scatter(x, y)

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')

plt.xlabel('Size of bitmap array')
plt.ylabel('False Positive Rate')

plt.savefig("correlation.png")

# Try to save the set to a hashmap and compare the size
hashTable = {}
for url in urllist:
    hashTable[url] = 1

hashtable_size = sys.getsizeof(hashTable)
bitmap_size = memory_usage_list[11]  # element 11 is chosen here because the fp rate is below 0.01 with this r size

with open("result_hashtable_bitmap.txt", "w") as txt_file:
    txt_file.write("HashTable size: " + "\n")
    txt_file.write(str(hashtable_size) + "\n")
    txt_file.write("BitMap size: " + "\n" + str(bitmap_size))
