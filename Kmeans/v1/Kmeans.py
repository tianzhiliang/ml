import sys
import os
import time
import math
import random

SLOT_SPLIT = "\t"
DATA_SPLIT = " "

class Tool:
    similarity_type = ""

    @staticmethod
    def float_equal(a, b):
        return math.fabs(a - b) < 1e-8
    
    @staticmethod
    def sign(a):
        if a >= 0:
            return 1
        else:
            return -1

    @staticmethod
    def cos_sim(a, b, len):
        aa = 0
        bb = 0
        ab = 0
        for i in range(len):
            aa += a[i] ** 2
            bb += b[i] ** 2
            ab += a[i] * b[i]
        
        aabb = aa * bb
        if Tool.float_equal(aabb, 0):
            return 0.0
        else:
            return ab * 1.0 / math.sqrt(aabb)
        
    @staticmethod
    def euc_dis(a, b, len):
        abs = 0
        for i in range(len):
            abs += (a[i] - b[i]) ** 2
        return math.sqrt(abs)

    @staticmethod
    def dot_product(a, b, len):
        ab = 0
        for i in range(len):
            ab += a[i] * b[i]
        return ab
    
    @staticmethod
    def similarity(a, b, len):
        if "cos" == Tool.similarity_type:
            return Tool.cos_sim(a, b, len)
        elif "euc" == Tool.similarity_type:
            return Tool.euc_dis(a, b, len)
        elif "dot_product" == Tool.similarity_type:
            return Tool.dot_product(a, b, len)
        elif "binary_dot_product" == Tool.similarity_type:
            return Tool.dot_product(a, b, len)
        else:
            print >> sys.stderr, "similarity type error in similarity"
     
    @staticmethod
    def find_mindis_and_index(list):
        if "cos" == Tool.similarity_type:
            mindis_value = max(list)
        elif "euc" == Tool.similarity_type:
            mindis_value = min(list)
        elif "dot_product" == Tool.similarity_type:
            mindis_value = max(list)
        elif "binary_dot_product" == Tool.similarity_type:
            mindis_value = max(list)

        mindis_index = list.index(mindis_value) # NOTICE: return the first mindis_value's index
        return mindis_value, mindis_index
            
class Sample:
    def __init__(self):
        self.sign = ""
        self.data = []
        self.category = 0 # category id
        self.distance = 0 # distance with category center(mean)
        self.mean = [] # category center, maybe no use

    def set(self, sign, data):
        self.sign = sign
        self.data = data

    def update(self, category, distance):
        self.category = category
        self.distance = distance
        
class Category:
    def __init__(self):
        self.mean = []
        self.samples_id = []
        self.samples_distance = []

    def clear(self):
        self.samples_id = []
        self.samples_distance = []

    def add_sample(self, sample_id):
        self.samples_id.append(sample_id)

    def add_sample(self, sample_id, distance):
        self.samples_id.append(sample_id)
        self.samples_distance.append(distance)

class Cluster:
    def __init__(self, category_num, similarity_type, total_iter):
        # data and parameter
        self.samples = []
        self.categories = []

        # variable
        self.dim = 0
        self.size_of_samples = 0
        self.iter = 0
        
        # config
        self.category_num = int(category_num) # K of K-means, eg: 2
        self.similarity_type = similarity_type # cos / binary_dot_product eg: dot_product
        self.total_iter = int(total_iter) # eg: 10

    def update_category(self):
        #clear samples in category
        for j in range(self.category_num): 
            self.categories[j].clear()

        for i in range(self.size_of_samples):
            # find category 
            dis_list = []
            for category in self.categories:
                sim = Tool.similarity(self.samples[i].data, category.mean, self.dim)
                dis_list.append(sim)
            distance, category_id = Tool.find_mindis_and_index(dis_list)

            # update sample
            self.samples[i].update(category_id, distance)

            # update category
            self.categories[category_id].add_sample(i, distance)
        print >> sys.stderr, time.ctime(), "TRACE, iter", self.iter, "update_category done"
            
    def update_means(self):
        for i in range(self.category_num):
            self.update_mean(i)
        print >> sys.stderr, time.ctime(), "TRACE, iter", self.iter, "update_means done"

    def update_mean(self, category_id): # using yinyong !!!!
        self.categories[category_id].mean = [0.0 for k in range(self.dim)]

        for sample_id in self.categories[category_id].samples_id:
            for i in range(self.dim):
                self.categories[category_id].mean[i] += self.samples[sample_id].data[i]
            
        if "binary_dot_product" == Tool.similarity_type:
            for i in range(self.dim):
                self.categories[category_id].mean[i] = Tool.sign(mean[i])
        
        print >> sys.stderr, time.ctime(), "DEBUG, in", self.iter, "iter, category:",  \
                 category_id, "means:", self.categories[category_id].mean,  \
                 "samples:", self.categories[category_id].samples_id, \
                 "distance:", self.categories[category_id].samples_distance

    def print_loss(self):
        distance = []
        for i in range(self.category_num):
            category_size = len(self.categories[i].samples_id)
            distance.append(sum(self.categories[i].samples_distance))

            if 0 != category_size:
                print >> sys.stderr, time.ctime(), "NOTICE, in", self.iter, "iter, category:", i, \
                    "sample_num:", category_size, "category_self_distance:", distance[i], "avg_distance:", \
                    distance[i] * 1.0 / category_size
        print >> sys.stderr, time.ctime(), "NOTICE, in", self.iter, "iter, total_distance:", \
                sum(distance), "avg_total_distance:", sum(distance) * 1.0 / self.size_of_samples
                 
    def init_means(self):
        random_id_set = {}
        for i in range(self.category_num):
            while True:
                random_id = random.randint(0, self.size_of_samples - 1)
                if random_id not in random_id_set:
                    random_id_set[random_id] = 1
                    break
            category = Category()
            category.mean = self.samples[random_id].data # random select sample as cluster center
            print >> sys.stderr, time.ctime(), "DEBUG, init_means, category:", i, "init:", self.samples[random_id].data
            self.categories.append(category)
    
    def init(self):
        Tool.similarity_type = self.similarity_type
        self.init_means()
        print >> sys.stderr, time.ctime(), "NOTICE, init done"
        
    def print_result(self):
        print >> sys.stderr, time.ctime(), "NOTICE, cluster done, will print_result"
        print >> sys.stderr, "sign" + SLOT_SPLIT + "data" + SLOT_SPLIT + "category_id" + SLOT_SPLIT + "distance"
        for sample in self.samples:
            data_str = map(str, sample.data)
            print sample.sign + SLOT_SPLIT + DATA_SPLIT.join(data_str) + SLOT_SPLIT + str(sample.category) + SLOT_SPLIT + str(sample.distance)
        print
            
        count = 0
        for category in self.categories:
            print "category ", count
            for j in category.samples_id:
                sample = self.samples[j]
                data_str = map(str, sample.data)
                print sample.sign + SLOT_SPLIT + DATA_SPLIT.join(data_str) + SLOT_SPLIT + str(sample.category) + SLOT_SPLIT + str(sample.distance)
            print
            count += 1
        print >> sys.stderr, time.ctime(), "NOTICE, will print_result done, totally done!!!"
            
    def load_sign_and_data(self, file):
        f = open(file, "r")
        print >> sys.stderr, time.ctime(), "NOTICE, open inputfile", file

        is_first_line = True
        for line in f:
            line = line.strip()
            slots = line.split(SLOT_SPLIT)
            sample = Sample()
            sample.sign = slots[0]
            sample.data = slots[1].split(DATA_SPLIT)
            if is_first_line:
                is_first_line = False
                self.dim = len(sample.data)
            elif self.dim != len(sample.data):
                print >> sys.stderr, time.ctime(), "ERROR, self.dim != len(sample.data)", self.dim, "!=", "len(sample.data), continue..."
                continue

            sample.data = map(float, sample.data)
            self.samples.append(sample)

        f.close()

        self.size_of_samples = len(self.samples)
        print >> sys.stderr, time.ctime(), "NOTICE, load_sign_and_data done, size_of_samples:", self.size_of_samples, "dim:", self.dim
        
    def run_K_means(self):
        self.init()
        for self.iter in range(self.total_iter):
            self.update_category()
            self.update_means()
            self.print_loss()
        self.print_result()

def main(argv):
    if 5 != len(argv):
        print >> sys.stderr, time.ctime(), "Usage:", argv[0], "inputfile category_num, similarity_type, total_iter"
        print >> sys.stderr, time.ctime(),  \
                "NOTICE, inputfile format: sample_sign <tab> sample_data_dim0 <space> sample_data_dim1 <space> ... <space> sample_data_dimN"
        exit(1)
    
    inputfile, category_num, similarity_type, total_iter = argv[1], argv[2], argv[3], argv[4]
    
    cluster = Cluster(category_num, similarity_type, total_iter)
    cluster.load_sign_and_data(inputfile)
    cluster.run_K_means()

if "__main__" == __name__:
    main(sys.argv)
