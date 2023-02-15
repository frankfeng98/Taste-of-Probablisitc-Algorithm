import sys
from matplotlib import pyplot as plt
from sklearn.utils import murmurhash3_32
import random
import heapq


class Min_Medium_Sketch():
    def __init__(self, d, r):
        # Size of the array (round up to the nearest power of 2
        self.size = r

        # Create d number of arrays
        self.array_of_arrays = []
        for i in range(d):
            self.array_of_arrays.append([0] * r)

        # Initialize the number of hash functions
        self.hash_number = d

        # Initialize the heap
        self.heap_min = {}
        self.heap_medium = {}

    def insert(self, key):
        for i in range(self.hash_number):
            pos = murmurhash3_32(key, i) % self.size
            self.array_of_arrays[i][pos] += 1
        # Maintain a min-heap of 500 size
        self.heap_min[key] = self.query_min(key)
        self.heap_medium[key] = self.query_medium(key)

    def query_min(self, key):
        least_count = sys.maxsize
        for i in range(self.hash_number):
            pos = murmurhash3_32(key, i) % self.size
            if self.array_of_arrays[i][pos] < least_count:
                least_count = self.array_of_arrays[i][pos]
        return least_count

    def query_medium(self, key):
        count = []
        for i in range(self.hash_number):
            pos = murmurhash3_32(key, i) % self.size
            count.append(self.array_of_arrays[i][pos])
        count.sort()
        return count[3]

    def get_min_dict(self):
        return heapq.nlargest(500, self.heap_min, key=self.heap_min.__getitem__)

    def get_medium_dict(self):
        return heapq.nlargest(500, self.heap_medium, key=self.heap_medium.__getitem__)

class Count_Sketch():
    def __init__(self, d, r):
        # Size of the array (round up to the nearest power of 2
        self.size = r

        # Create d number of arrays
        self.array_of_arrays = []
        for i in range(d):
            self.array_of_arrays.append([0] * r)

        # Initialize the number of hash functions
        self.hash_number = d

        # Initialize the sign array
        self.sign_set = [-1, 1]

        # Initialize the heap
        self.heap = {}

    def insert(self, key):
        for i in range(self.hash_number):
            pos = murmurhash3_32(key, i) % self.size
            sign_index = murmurhash3_32(key, i + 10) % 2
            sign = self.sign_set[sign_index - 1]
            self.array_of_arrays[i][pos] += 1 * sign
        # Maintain a min-heap of 500 size
        self.heap[key] = self.query(key)

    def query(self, key):
        count = []
        for i in range(self.hash_number):
            pos = murmurhash3_32(key, i) % self.size
            sign_index = murmurhash3_32(key, i + 10) % 2
            sign = self.sign_set[sign_index - 1]
            count.append(self.array_of_arrays[i][pos] * sign)
        count.sort()
        return count[3]

    def get_dict(self):
        return heapq.nlargest(500, self.heap, key=self.heap.__getitem__)

def main():
    # Create appropriate sketch methods and the dictionary
    min_medium_sketch_small = Min_Medium_Sketch(5, pow(2, 10))
    min_medium_sketch_medium = Min_Medium_Sketch(5, pow(2, 14))
    min_medium_sketch_big = Min_Medium_Sketch(5, pow(2, 18))
    count_sketch_small = Count_Sketch(5, pow(2, 10))
    count_sketch_medium = Count_Sketch(5, pow(2, 14))
    count_sketch_big = Count_Sketch(5, pow(2, 18))
    tracker = {}

    data = open('user-ct-test-collection-01.txt')
    data.readline()
    for line in data:
        contents = line.split()
        index = 1
        # Traverse through the line until date field has been reached (exclude any special characters)
        while True:
            if "2006-" in contents[index]:
                break
            print(contents[index])
            # Do something here with the contents[index]
            min_medium_sketch_small.insert(contents[index])
            min_medium_sketch_medium.insert(contents[index])
            min_medium_sketch_big.insert(contents[index])
            count_sketch_small.insert(contents[index])
            count_sketch_medium.insert(contents[index])
            count_sketch_big.insert(contents[index])
            tracker[contents[index]] = tracker.get(contents[index], 1) + 1

            # Update the index to retrieve the next word
            index += 1

    # Get the top 100, bottom 100 elements and random 100 elements
    print("Sort started")
    largest_100 = heapq.nlargest(100, tracker, key=tracker.__getitem__)
    smallest_100 = heapq.nsmallest(100, tracker, key=tracker.__getitem__)
    random.seed(100)
    random_100_tuple = random.sample(tracker.items(), 100)
    random_100 = dict((x, y) for x, y in random_100_tuple)
    random_100_sorted = heapq.nlargest(100, random_100, key=random_100.__getitem__)

    # Begin to calculate the error rate for largest 100
    largest_100_min_large_er = []
    largest_100_medium_large_er = []
    largest_100_count_large_er = []
    largest_100_min_medium_er = []
    largest_100_medium_medium_er = []
    largest_100_count_medium_er = []
    largest_100_min_small_er = []
    largest_100_medium_small_er = []
    largest_100_count_small_er = []
    for word in largest_100:
        actual_count = tracker.get(word)
        min_large_count = min_medium_sketch_big.query_min(word)
        min_medium_count = min_medium_sketch_medium.query_min(word)
        min_small_count = min_medium_sketch_small.query_min(word)
        medium_large_count = min_medium_sketch_big.query_medium(word)
        medium_medium_count = min_medium_sketch_medium.query_medium(word)
        medium_small_count = min_medium_sketch_small.query_medium(word)
        count_large_count = count_sketch_big.query(word)
        count_medium_count = count_sketch_medium.query(word)
        count_small_count = count_sketch_small.query(word)

        # Update the error rate for each method
        largest_100_min_large_er.append(abs(min_large_count - actual_count) / actual_count)
        largest_100_min_medium_er.append(abs(min_medium_count - actual_count) / actual_count)
        largest_100_min_small_er.append(abs(min_small_count - actual_count) / actual_count)
        largest_100_medium_large_er.append(abs(medium_large_count - actual_count) / actual_count)
        largest_100_medium_medium_er.append(abs(medium_medium_count - actual_count) / actual_count)
        largest_100_medium_small_er.append(abs(medium_small_count - actual_count) / actual_count)
        largest_100_count_large_er.append(abs(count_large_count - actual_count) / actual_count)
        largest_100_count_medium_er.append(abs(count_medium_count - actual_count) / actual_count)
        largest_100_count_small_er.append(abs(count_small_count - actual_count) / actual_count)

    # Draw the 3 graphs for largest 100 elements
    # Draw the graph for smallest range of R
    plt.plot(largest_100, largest_100_min_small_er, label="Min Sketch", linestyle="-")
    plt.plot(largest_100, largest_100_medium_small_er, label="Medium Sketch", linestyle="--")
    plt.plot(largest_100, largest_100_count_small_er, label="Count Sketch", linestyle=":")
    plt.legend()
    plt.title("Small R Freq-100")
    plt.xlabel("Words by frequency")
    plt.ylabel("Error rate")
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("Small R Freq-100.png")

    # Draw the graph for medium range of R
    plt.figure()
    plt.plot(largest_100, largest_100_min_medium_er, label="Min Sketch", linestyle="-")
    plt.plot(largest_100, largest_100_medium_medium_er, label="Medium Sketch", linestyle="--")
    plt.plot(largest_100, largest_100_count_medium_er, label="Count Sketch", linestyle=":")
    plt.legend()
    plt.title("Medium R Freq-100")
    plt.xlabel("Words by frequency")
    plt.ylabel("Error rate")
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig("Medium R Freq-100.png")

    # Draw the graph for large range of R
    plt.figure()
    plt.plot(largest_100, largest_100_min_large_er, label="Min Sketch", linestyle="-")
    plt.plot(largest_100, largest_100_medium_large_er, label="Medium Sketch", linestyle="--")
    plt.plot(largest_100, largest_100_count_large_er, label="Count Sketch", linestyle=":")
    plt.legend()
    plt.title("Large R Freq-100")
    plt.xlabel("Words by frequency")
    plt.ylabel("Error rate")
    fig3 = plt.gcf()
    plt.show()
    fig3.savefig("Large R Freq-100.png")

    # Calculate error rate for the smallest 100 elements
    smallest_100_min_large_er = []
    smallest_100_medium_large_er = []
    smallest_100_count_large_er = []
    smallest_100_min_medium_er = []
    smallest_100_medium_medium_er = []
    smallest_100_count_medium_er = []
    smallest_100_min_small_er = []
    smallest_100_medium_small_er = []
    smallest_100_count_small_er = []
    # Get the most frequent values first
    for word_index in range(99, -1, -1):
        word = smallest_100[word_index]
        actual_count = tracker.get(word)
        min_large_count = min_medium_sketch_big.query_min(word)
        min_medium_count = min_medium_sketch_medium.query_min(word)
        min_small_count = min_medium_sketch_small.query_min(word)
        medium_large_count = min_medium_sketch_big.query_medium(word)
        medium_medium_count = min_medium_sketch_medium.query_medium(word)
        medium_small_count = min_medium_sketch_small.query_medium(word)
        count_large_count = count_sketch_big.query(word)
        count_medium_count = count_sketch_medium.query(word)
        count_small_count = count_sketch_small.query(word)

        # update the error rate for each method
        smallest_100_min_large_er.append(abs(min_large_count - actual_count) / actual_count)
        smallest_100_min_medium_er.append(abs(min_medium_count - actual_count) / actual_count)
        smallest_100_min_small_er.append(abs(min_small_count - actual_count) / actual_count)
        smallest_100_medium_large_er.append(abs(medium_large_count - actual_count) / actual_count)
        smallest_100_medium_medium_er.append(abs(medium_medium_count - actual_count) / actual_count)
        smallest_100_medium_small_er.append(abs(medium_small_count - actual_count) / actual_count)
        smallest_100_count_large_er.append(abs(count_large_count - actual_count) / actual_count)
        smallest_100_count_medium_er.append(abs(count_medium_count - actual_count) / actual_count)
        smallest_100_count_small_er.append(abs(count_small_count - actual_count) / actual_count)

    # Draw the 3 graphs for smallest 100 elements
    # Draw the graph for smallest range of R
    plt.plot(smallest_100, smallest_100_min_small_er, label="Min Sketch", linestyle="-")
    plt.plot(smallest_100, smallest_100_medium_small_er, label="Medium Sketch", linestyle="--")
    plt.plot(smallest_100, smallest_100_count_small_er, label="Count Sketch", linestyle=":")
    plt.legend()
    plt.title("Small R Infreq-100")
    plt.xlabel("Words by frequency")
    plt.ylabel("Error rate")
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("Small R Infreq-100.png")

    # Draw the graph for medium range of R
    plt.figure()
    plt.plot(smallest_100, smallest_100_min_medium_er, label="Min Sketch", linestyle="-")
    plt.plot(smallest_100, smallest_100_medium_medium_er, label="Medium Sketch", linestyle="--")
    plt.plot(smallest_100, smallest_100_count_medium_er, label="Count Sketch", linestyle=":")
    plt.legend()
    plt.title("Medium R Infreq-100")
    plt.xlabel("Words by frequency")
    plt.ylabel("Error rate")
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig("Medium R Infreq-100.png")

    # Draw the graph for large range of R
    plt.figure()
    plt.plot(smallest_100, smallest_100_min_large_er, label="Min Sketch", linestyle="-")
    plt.plot(smallest_100, smallest_100_medium_large_er, label="Medium Sketch", linestyle="--")
    plt.plot(smallest_100, smallest_100_count_large_er, label="Count Sketch", linestyle=":")
    plt.legend()
    plt.title("Large R Infreq-100")
    plt.xlabel("Words by frequency")
    plt.ylabel("Error rate")
    fig3 = plt.gcf()
    plt.show()
    fig3.savefig("Large R Infreq-100.png")

    # Calculate error rate for the random 100 elements
    random_100_min_large_er = []
    random_100_medium_large_er = []
    random_100_count_large_er = []
    random_100_min_medium_er = []
    random_100_medium_medium_er = []
    random_100_count_medium_er = []
    random_100_min_small_er = []
    random_100_medium_small_er = []
    random_100_count_small_er = []
    for word in random_100_sorted:
        actual_count = tracker.get(word)
        min_large_count = min_medium_sketch_big.query_min(word)
        min_medium_count = min_medium_sketch_medium.query_min(word)
        min_small_count = min_medium_sketch_small.query_min(word)
        medium_large_count = min_medium_sketch_big.query_medium(word)
        medium_medium_count = min_medium_sketch_medium.query_medium(word)
        medium_small_count = min_medium_sketch_small.query_medium(word)
        count_large_count = count_sketch_big.query(word)
        count_medium_count = count_sketch_medium.query(word)
        count_small_count = count_sketch_small.query(word)

        # update the error rate for each method
        random_100_min_large_er.append(abs(min_large_count - actual_count) / actual_count)
        random_100_min_medium_er.append(abs(min_medium_count - actual_count) / actual_count)
        random_100_min_small_er.append(abs(min_small_count - actual_count) / actual_count)
        random_100_medium_large_er.append(abs(medium_large_count - actual_count) / actual_count)
        random_100_medium_medium_er.append(abs(medium_medium_count - actual_count) / actual_count)
        random_100_medium_small_er.append(abs(medium_small_count - actual_count) / actual_count)
        random_100_count_large_er.append(abs(count_large_count - actual_count) / actual_count)
        random_100_count_medium_er.append(abs(count_medium_count - actual_count) / actual_count)
        random_100_count_small_er.append(abs(count_small_count - actual_count) / actual_count)

    # Draw the 3 graphs for Random 100 elements
    # Draw the graph for smallest range of R
    plt.plot(random_100_sorted, random_100_min_small_er, label="Min Sketch", linestyle="-")
    plt.plot(random_100_sorted, random_100_medium_small_er, label="Medium Sketch", linestyle="--")
    plt.plot(random_100_sorted, random_100_count_small_er, label="Count Sketch", linestyle=":")
    plt.legend()
    plt.title("Smallest R Random-100")
    plt.xlabel("Words by frequency")
    plt.ylabel("Error rate")
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("Small R Random-100.png")

    # Draw the graph for medium range of R
    plt.figure()
    plt.plot(random_100_sorted, random_100_min_medium_er, label="Min Sketch", linestyle="-")
    plt.plot(random_100_sorted, random_100_medium_medium_er, label="Medium Sketch", linestyle="--")
    plt.plot(random_100_sorted, random_100_count_medium_er, label="Count Sketch", linestyle=":")
    plt.legend()
    plt.title("Medium R Random-100")
    plt.xlabel("Words by frequency")
    plt.ylabel("Error rate")
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig("Medium R Random-100.png")

    # Draw the graph for small range of R
    plt.figure()
    plt.plot(random_100_sorted, random_100_min_large_er, label="Min Sketch", linestyle="-")
    plt.plot(random_100_sorted, random_100_medium_large_er, label="Medium Sketch", linestyle="--")
    plt.plot(random_100_sorted, random_100_count_large_er, label="Count Sketch", linestyle=":")
    plt.legend()
    plt.title("Large R Random-100")
    plt.xlabel("Words by frequency")
    plt.ylabel("Error rate")
    fig3 = plt.gcf()
    plt.show()
    fig3.savefig("Large R Random-100.png")

    # Calculate the intersection between Top 500 heap and Top 100 elements
    Min_sketch_dictionary_small = min_medium_sketch_small.get_min_dict()
    Min_sketch_dictionary_medium = min_medium_sketch_medium.get_min_dict()
    Min_sketch_dictionary_large = min_medium_sketch_big.get_min_dict()
    min_intersction_count_small = 0
    min_intersction_count_medium = 0
    min_intersction_count_large = 0
    intersection_minSketch = []

    Medium_sketch_dictionary_small = min_medium_sketch_small.get_medium_dict()
    Medium_sketch_dictionary_medium = min_medium_sketch_medium.get_medium_dict()
    Medium_sketch_dictionary_large = min_medium_sketch_big.get_medium_dict()
    medium_intersction_count_small = 0
    medium_intersction_count_medium = 0
    medium_intersction_count_large = 0
    intersection_mediumSketch = []

    count_sketch_dictionary_small = count_sketch_small.get_dict()
    count_sketch_dictionary_medium = count_sketch_medium.get_dict()
    count_sketch_dictionary_large = count_sketch_big.get_dict()
    count_intersction_count_small = 0
    count_intersction_count_medium = 0
    count_intersction_count_large = 0
    intersection_countSketch = []

    for word in largest_100:
        if word in Min_sketch_dictionary_small:
            min_intersction_count_small += 1
        if word in Min_sketch_dictionary_medium:
            min_intersction_count_medium += 1
        if word in Min_sketch_dictionary_large:
            min_intersction_count_large += 1
        if word in count_sketch_dictionary_small:
            count_intersction_count_small += 1
        if word in count_sketch_dictionary_medium:
            count_intersction_count_medium += 1
        if word in count_sketch_dictionary_large:
            count_intersction_count_large += 1
        if word in Medium_sketch_dictionary_small:
            medium_intersction_count_small += 1
        if word in Medium_sketch_dictionary_medium:
            medium_intersction_count_medium += 1
        if word in Medium_sketch_dictionary_large:
            medium_intersction_count_large += 1

    intersection_minSketch.append(min_intersction_count_small)
    intersection_minSketch.append(min_intersction_count_medium)
    intersection_minSketch.append(min_intersction_count_large)

    intersection_mediumSketch.append(medium_intersction_count_small)
    intersection_mediumSketch.append(medium_intersction_count_medium)
    intersection_mediumSketch.append(medium_intersction_count_large)

    intersection_countSketch.append(count_intersction_count_small)
    intersection_countSketch.append(count_intersction_count_medium)
    intersection_countSketch.append(count_intersction_count_large)

    # Draw the graph for intersections
    x = ["2^10", "2^14", "2^18"]
    plt.figure()
    plt.plot(x, intersection_minSketch, label="Min Sketch", linestyle="-")
    plt.legend()
    plt.title("Intersection between Min sketch TOP 500 and Freq-100")
    plt.xlabel("R size")
    plt.ylabel("Intersection size")
    fig3 = plt.gcf()
    plt.show()
    fig3.savefig("Intersection between Min sketch TOP 500 and Freq-100.png")

    plt.figure()
    plt.plot(x, intersection_mediumSketch, label="Medium Sketch", linestyle="-")
    plt.legend()
    plt.title("Intersection between Medium sketch TOP 500 and Freq-100")
    plt.xlabel("R size")
    plt.ylabel("Intersection Size")
    fig3 = plt.gcf()
    plt.show()
    fig3.savefig("Intersection between Medium sketch TOP 500 and Freq-100.png")

    plt.figure()
    plt.plot(x, intersection_countSketch, label="Count Sketch", linestyle="-")
    plt.legend()
    plt.title("Intersection between count sketch TOP 500 and Freq-100")
    plt.xlabel("R size")
    plt.ylabel("Intersection size")
    fig3 = plt.gcf()
    plt.show()
    fig3.savefig("Intersection between count sketch TOP 500 and Freq-100.png")


main()
