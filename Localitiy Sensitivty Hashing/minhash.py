import sys
import pandas as pd
from sklearn.utils import murmurhash3_32
from matplotlib import pyplot as plt
import random
import time
import heapq


# minHash generator
def MinHash(input_string, hashcode_num):
    minhash_values = []
    input_subsets = []
    # Generate subsets based on 3-gram representation
    for index in range(0, len(input_string) - 2, 1):
        temp_string = input_string[index:index + 3]
        input_subsets.append(temp_string)

    # Hash the subsets of string using m numbers of murmurhash3 & store min hash values
    for num in range(hashcode_num):
        min_num = sys.maxsize
        for subset in input_subsets:
            hash_result = murmurhash3_32(subset, num)
            if hash_result < min_num:
                min_num = hash_result
        minhash_values.append(min_num)

    return minhash_values


class HashTable():
    def __init__(self, K, L, B, R):
        self.hash_num = K
        self.table_num = L
        self.hash_range = R
        self.random_seeds = K * L
        self.current_seed = self.random_seeds
        self.hash_tables = [{} for _ in range(self.hash_num)]

    def insert(self, hashcodes, id):
        random.seed(self.random_seeds)
        for hash_table in self.hash_tables:
            sum_for_hash = 0
            p = pow(2, 25)
            exponents = len(hashcodes)
            for hashcode in hashcodes:
                a = random.randrange(1, p)
                sum_for_hash += a * pow(hashcode, exponents)
                exponents -= 1
            c = random.randrange(1, p)
            sum_for_hash += c
            position = sum_for_hash % p % self.hash_range
            current_list_or_none = hash_table.get(position)
            if current_list_or_none is None:
                current_list_or_none = [id]
            else:
                current_list_or_none.append(id)
            hash_table[position] = current_list_or_none

    def lookup(self, hashcodes):
        random.seed(self.random_seeds)
        union_buckets = []
        for hash_table in self.hash_tables:
            sum_for_hash = 0
            p = pow(2, 25)
            exponents = len(hashcodes)
            for hashcode in hashcodes:
                a = random.randrange(1, p)
                sum_for_hash += a * pow(hashcode, exponents)
                exponents -= 1
            c = random.randrange(1, p)
            sum_for_hash += c
            position = sum_for_hash % p % self.hash_range
            current_list = hash_table.get(position)
            union_buckets.append(current_list)
        return union_buckets


# Test two input strings
def naive_test():
    print("Testing minHash generators with two input strings...")
    input_1 = "The mission statement of the WCSCC and area employers recognize the importance of good attendance on " \
              "the job. Any student whose absences exceed 18 days is jeopardizing their opportunity for advanced " \
              "placement as well as hindering his/her likelihood for successfully completing their program."
    input_2 = "The WCSCC’s mission statement and surrounding employers recognize the importance of great attendance. " \
              "Any student who is absent more than 18 days will loose the opportunity for successfully completing " \
              "their trade program."
    print("Jaccard Similarity is: " + str(jaccard_similarity(input_1, input_2)))

    # Calculate the similarity based on minhash values
    minhash_value_1 = MinHash(input_1, 100)
    minhash_value_2 = MinHash(input_2, 100)
    intersection_minhash = len(list(set(minhash_value_1).intersection(minhash_value_2)))
    minhash_similarity = intersection_minhash / 100
    print("Minhash estimated similarity is: " + str(minhash_similarity))


def jaccard_similarity(input_1, input_2):
    input_1_substrings = []
    input_2_substrings = []

    # Decompose string into 3-gram representations
    for index in range(0, len(input_1) - 2, 1):
        temp_string = input_1[index:index + 3]
        input_1_substrings.append(temp_string)
    for index in range(0, len(input_2) - 2, 1):
        temp_string = input_2[index:index + 3]
        input_2_substrings.append(temp_string)

    # Calculate the Jaccard Similaritiy
    intersection_jaccard = len(list(set(input_1_substrings).intersection(input_2_substrings)))
    union_jaccard = (len(input_1_substrings) + len(input_2_substrings)) - intersection_jaccard
    jaccard_similarity_result = intersection_jaccard / union_jaccard
    return jaccard_similarity_result


def task_1(urllist, sample_urls):
    print("Task 1 begins...")
    print("Inserting urls into hashtables...")
    hash_table_1 = HashTable(2, 50, 64, pow(2, 20))
    for url in urllist:
        url_hashcodes = MinHash(url, 2)
        hash_table_1.insert(url_hashcodes, url)

    print("Query begins")
    task1_mean_jaccard_similarity_sum = 0
    task1_mean_jaccard_similarity_top10_sum = 0
    start_time = time.time()
    count = 0
    for sample_url in sample_urls:
        print("Sample" + sample_url)
        print(str(count))
        count += 1
        sample_hashcodes = MinHash(sample_url, 2)
        candidate_set_set = hash_table_1.lookup(sample_hashcodes)
        jaccard_similarity_each_url = []
        for candidate_set in candidate_set_set:
            for candidate in candidate_set:
                js_candidate = jaccard_similarity(sample_url, candidate)
                jaccard_similarity_each_url.append(js_candidate)
        task1_mean_jaccard_similarity_sum += sum(jaccard_similarity_each_url) / len(jaccard_similarity_each_url)
        top10_url = heapq.nlargest(10, jaccard_similarity_each_url)
        task1_mean_jaccard_similarity_top10_sum += sum(top10_url) / len(top10_url)
    end_time = time.time()
    print("Query time = " + str((end_time - start_time)))
    print("Quality of all candidate urls: ")
    print(task1_mean_jaccard_similarity_sum / 200)
    print("Quality of top 10 candidate urls: ")
    print(task1_mean_jaccard_similarity_top10_sum / 200)


def task_2(urllist, sample_urls):
    print("Task 2 begins...")
    start_time = time.time()
    for sample_url in sample_urls:
        for url in urllist:
            jaccard_similarity(sample_url, url)
    end_time = time.time()
    print("Time of brute-force way of calculating: " + str(end_time - start_time))


def task_3(urllist, sample_urls):
    print("Task 3 begins...")
    K_option = [2, 3, 4, 5, 6]
    L_option = [20, 50, 100]
    for K in K_option:
        if K == 2:
            sample_urls_used = sample_urls[0: 20]
        else:
            sample_urls_used = sample_urls

        for L in L_option:
            print("K is " + str(K))
            print("L is " + str(L))
            # Insert urls into hashtable
            hash_table = HashTable(K, L, 64, pow(2, 20))
            for url in urllist:
                url_hashcodes = MinHash(url, K)
                hash_table.insert(url_hashcodes, url)

            # Query the urls
            task3_mean_jaccard_similarity_sum = 0
            task3_mean_jaccard_similarity_top10_sum = 0
            start_time = time.time()
            for sample_url in sample_urls_used:
                sample_hashcodes = MinHash(sample_url, K)
                candidate_set_set = hash_table.lookup(sample_hashcodes)
                jaccard_similarity_each_url = []
                for candidate_set in candidate_set_set:
                    for candidate in candidate_set:
                        js_candidate = jaccard_similarity(sample_url, candidate)
                        jaccard_similarity_each_url.append(js_candidate)
                task3_mean_jaccard_similarity_sum += sum(jaccard_similarity_each_url) / len(jaccard_similarity_each_url)
                top10_url = heapq.nlargest(10, jaccard_similarity_each_url)
                task3_mean_jaccard_similarity_top10_sum += sum(top10_url) / len(top10_url)
            end_time = time.time()
            print("Query time = " + str((end_time - start_time)))
            print("Quality of all candidate urls: ")
            if K == 2:
                print(task3_mean_jaccard_similarity_sum / 20)
                print("Quality of top 10 candidate urls: ")
                print(task3_mean_jaccard_similarity_top10_sum / 20)
            else:
                print(task3_mean_jaccard_similarity_sum / 200)
                print("Quality of top 10 candidate urls: ")
                print(task3_mean_jaccard_similarity_top10_sum / 200)

def task_4():
    print("Task 4 begins...")
    print("Calculating reference Jaccard Similarity for the task...")
    jaccard_similarity_one_element = []
    element = 0
    while element <= 1:
        jaccard_similarity_one_element.append(element)
        element += 0.00001

    print("Drawing plot 1...")
    print("Calculating possibility for p...")
    K_option = [1, 2, 3, 4, 5, 6, 7]
    L = 50
    probability_of_retrieveing = [[] for _ in range(7)]
    for index in range(7):
        for js in jaccard_similarity_one_element:
            probaility = 1 - pow((1 - js ** K_option[index]), L)
            probability_of_retrieveing[index].append(probaility)
    # Start drawing graph
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[0], label="K = 1")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[1], label="K = 2")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[2], label="K = 3")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[3], label="K = 4")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[4], label="K = 5")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[5], label="K = 6")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[6], label="K = 7")
    plt.legend()
    plt.title("Plot 1 with fixed L")
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Probability of retrieving element x")
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("Plot 1 with fixed L.png")

    print("Drawing plot 2...")
    print("Calculating possibility for p...")
    K = 4
    L_option = [5, 10, 20, 50, 100, 150, 200]
    probability_of_retrieveing = [[] for _ in range(7)]
    for index in range(7):
        for js in jaccard_similarity_one_element:
            probaility = 1 - pow((1 - js ** K), L_option[index])
            probability_of_retrieveing[index].append(probaility)
    # Start drawing graph
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[0], label="L = 5")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[1], label="L = 10")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[2], label="L = 20")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[3], label="L = 50")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[4], label="L = 100")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[5], label="L = 150")
    plt.plot(jaccard_similarity_one_element, probability_of_retrieveing[6], label="L = 200")
    plt.legend()
    plt.title("Plot 2 with fixed K")
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Probability of retrieving element x")
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("Plot 2 with fixed K.png")


def main():
    # Testing minHash algorithm with two input strings
    naive_test()

    # Data preprocessing
    data = pd.read_csv('user-ct-test-collection-01.txt', sep="\t")
    urllist = data.ClickURL.dropna().unique()
    random.seed(10086)
    sample_urls = random.sample(list(urllist), 200)

    # Task 1
    # Insert urls into hash tables
    task_1(urllist, sample_urls)

    # Task 2
    task_2(urllist, sample_urls)

    # Task 3
    task_3(urllist, sample_urls)

    # Task 4
    task_4()


main()