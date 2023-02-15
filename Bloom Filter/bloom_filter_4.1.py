from sklearn.utils import murmurhash3_32
from bitarray import bitarray
import math
import numpy as np
import random

def getSize(n, fp_rate):
    size = -(n * math.log(fp_rate)) / (math.log(2) ** 2)
    return int(size)


class BloomFilter():
    def __init__(self, n, fp_rate):
        # Size of the array (round up to the nearest power of 2
        self.size = getSize(n, fp_rate)

        # Initialize the array
        self.bit_array = bitarray(self.size)

        # Calculate the number of hash functions
        self.hash_number = math.ceil(np.log(2) * math.log(fp_rate, 0.618))

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


# WarmUp Test
# Generate 10,000 membership and 1000 test cases
random.seed(10)
testSet = random.sample(range(1, 9999), 1000)
random.seed(20)
membershipSet = random.sample(range(10000, 99999), 10000)
random.seed(30)
testSet_2_pos = random.sample(range(0, 9999), 1000)

output = [["Theoretical FP", "Real FP"], ["0.01", 0], ["0.001", 0], ["0.0001", 0]]

for pos in testSet_2_pos:
    testSet.append(membershipSet[pos])

# Test the 0.01 false positive rate scenario
testCase_1 = BloomFilter(10000, 0.01)
for key in range(10000):
    testCase_1.insert(membershipSet[key])

falseCount = 0
for case in range(2000):
    if testSet[case] < 10000:
        if testCase_1.test(testSet[case]):
            falseCount = falseCount + 1

fpRate = falseCount / 1000
output[1][1] = str(fpRate)

# Test the 0.001 false positive rate case
testCase_2 = BloomFilter(10000, 0.001)
for key_2 in range(10000):
    testCase_2.insert(membershipSet[key_2])

falseCount = 0
for case in range(2000):
    if testSet[case] < 10000:
        if testCase_2.test(testSet[case]):
            falseCount = falseCount + 1

fpRate = falseCount / 1000
output[2][1] = str(fpRate)

# Test the 0.0001 false positive rate case
testCase_3 = BloomFilter(10000, 0.0001)
for key_3 in range(10000):
    testCase_3.insert(membershipSet[key_3])

falseCount = 0
for case in range(2000):
    if testSet[case] < 10000:
        if testCase_3.test(testSet[case]):
            falseCount = falseCount + 1

fpRate = falseCount / 1000
output[3][1] = str(fpRate)

# Print the result and save it as text file
output_np = np.array(output)
print(output_np)
with open("result_new.txt", "w") as txt_file:
    for line in output:
        txt_file.write(" ".join(line) + "\n")

