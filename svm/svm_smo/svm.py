import os
import sys
import random
import time
import math
import numpy # TODO: add try catch, add branch: not using numpy

class Tool:
    @staticmethod
    def float_equal(a, b):
        return math.fabs(a - b) < 1e-8
        
    @staticmethod
    def dot_product(a, b):
        return numpy.dot(a, b) # TODO: add try catch, add branch: not using numpy
    
    @staticmethod
    def sign(a):
        if a >= 0:
            return 1
        else:
            return -1

class Config:
    def __init__(self):
#    def __init__(self, relaxation_factor, w_random_range, b_random_range, a_random_range, total_epoch, kernel_type):
        self.train_sample_num = 0
        self.predict_sample_num = 0
        self.feature_dim = 0
        self.relaxation_factor = 0.5
        self.w_random_range = 0 # (-w_random_range, w_random_range)
        self.b_random_range = 0.5 # b_random_range 
        self.a_random_range = 0
        self.total_epoch = 5
        self.kernel_type = 0
        self.update_score_batch = 32


class Model:
    def __init__(self):
        # model parameter, hyper-parameter, important variables
        self.a = [] # Lanrange multiplier
        self.E = [] # loss
        self.b = [] # bias parameter
        self.C = 0 # relaxation_factor
        self.W = [] # weight parameter
        self.config = None
        self.dim = 0
        self.N = 0
        self.total_epoch = 0
        self.kernel_type = 0
        self.update_score_batch = 0

        # data
        self.x = [] # samples' input features, a matrix
        self.y = [] # samples' label (ground-turth)

        # temp variable
        self.g = [] # score of each sample, g(x) = \sigma (w_i * x_i) _ b
        self.support_vector_dict = {}

        # sign and stat
        self.SV_is_updated = True # support vector list have been updated or not, since the lastest read
        self.trained_sample = 0
        self.traind_epoch = 0

    def set_config_and_init(self, config):
        self.config = config
        self.b = (random.random() - 1) * 2 * config.b_random_range
        #self.a = [(random.random() - 1) * 2 * a_random_range for i in range(config.train_sample_num)] # TODO ??? 
        self.a = [0 for i in range(config.train_sample_num)]
        self.E = [0 for i in range(config.train_sample_num)]
        self.W = [0 for i in range(config.feature_dim)]
        #self.W = [(random.random() - 1) * 2 * w_random_range for i in range(config.feature_dim)]
        self.g = [0 for i in range(config.train_sample_num)]
        self.C = config.relaxation_factor
        self.dim = config.feature_dim
        self.N = config.train_sample_num
        self.total_epoch = config.total_epoch
        self.kernel_type = config.kernel_type
        self.update_score_batch = config.update_score_batch
        print >> sys.stderr, time.ctime(), "INFO, set_config_and_init done, train_sample_num:", self.N, "feature_dim", self.dim, "relaxation_factor:", self.C, "total_epoch:", self.total_epoch, "kernel_type:", self.kernel_type, "b_random_range:", config.b_random_range

    def train_all_epoch(self):
        print >> sys.stderr, time.ctime(), "TRACE, begin train_all_epoch"
        self.init_res_error_on_traindata()

        for i in range(self.total_epoch):
            print >> sys.stderr, time.ctime(), "TRACE, begin train epoch:", i, ". Total epoch:", self.total_epoch
            self.train()
            self.trained_sample = 0
            self.traind_epoch += 1

    def is_support_vector(self, index):
        if 0 < self.a[index] < self.C: 
            return True
        else:
            return False
        
    def update_support_vectors(self):
        self.support_vector_dict = {}
        for i in range(self.N):
            if self.is_support_vector(i):
                self.support_vector_dict[i] = 1
        self.SV_is_updated = True
        print >> sys.stderr, time.ctime(), "INFO, update_support_vectors done, size:", len(self.support_vector_dict.keys())

    def get_support_vectors_list(self):
        print >> sys.stderr, time.ctime(), "INFO, get_support_vectors_list size:", len(self.support_vector_dict.keys())
        return self.support_vector_dict.keys() # take care: disorder

    def get_support_vectors_dict(self):
        print >> sys.stderr, time.ctime(), "INFO, get_support_vectors_dict size:", len(self.support_vector_dict.keys())
        return self.support_vector_dict
    
    def kernel(self, a_index, b_index):
        if 0 == self.kernel_type:
            return Tool.dot_product(self.x[a_index], self.x[b_index])
            # TODO: other kernel

    def square_sum_of_kernel(self, a_index, b_index):
        # Kaa + Kbb - 2 * Kab
        res = self.kernel(a_index, a_index) + self.kernel(b_index, b_index) - 2 * self.kernel(a_index, b_index)
        return res
    
    def get_score_during_training(self, index):
        g = self.b
        for i in range(self.N):
            g += self.a[i] * self.y[i] * self.kernel(i, index)
        return g

    def update_score_on_traindata(self): # watse time!!!
        print >> sys.stderr, time.ctime(), "DEBUG, update_score_on_traindata:"
        for i in range(self.N):
            self.g[i] = self.get_score_during_training(i)
            print >> sys.stderr, str(i) + "|" + str(self.y[i]) + "|" + str(self.g[i]),
        print >> sys.stderr, ""
        print >> sys.stderr, time.ctime(), "TRACE, update_score_on_traindata done"

    def get_accurcay(self):
        right = 0
        wrong = 0
        for i in range(self.N):
            pred = Tool.sign(self.g[i])
            if self.y[i] == pred:
                right += 1
            else:
                wrong += 1

        acc = right * 1.0 / (right + wrong)
        print >> sys.stderr, time.ctime(), "INFO, epoch", self.traind_epoch - 1, "samples in this epoch:", self.trained_sample, "accuracy:", acc
    
    def init_res_error_on_traindata(self): # only run at first, watse time!!!
        print >> sys.stderr, time.ctime(), "DEBUG, self.E[i]:"
        for i in range(self.N):
            self.E[i] = self.b - self.y[i]
            for j in range(self.N):
                self.E[i] += self.a[j] * self.y[j] * self.kernel(i, j)
            print >> sys.stderr, self.E[i],
        print >> sys.stderr, ""
        print >> sys.stderr, time.ctime(), "TRACE, init_res_error_on_traindata done"
    
    def is_satisfy_KKT(self, index):
        gy = self.g[index] * self.y[index]
        print >> sys.stderr, time.ctime(), "DEBUG, is_satisfy_KKT index:", index, "g:", self.g[index], "y:", self.y[index], "gy:", gy, "a:", self.a[index]

        if Tool.float_equal(self.a[index], 0):
            if gy < 1:
                return False
        elif 0 < self.a[index] < self.C:
            if not Tool.float_equal(gy, 1):
                return False
        elif Tool.float_equal(self.a[index], self.C):
            if gy > 1:
                return False
        else:
            print >> sys.stderr, time.ctime(), "ERROR, in is_satisfy_KKT"

        return True
    
    def customized_new_a(self, a1, a2, a2_new_value):
        if self.y[a1] != self.y[a2]:
            H = min([self.C, self.C + self.a[a2] - self.a[a1]])
            L = max([0, self.a[a2] - self.a[a1]])
        else:
            H = min([self.C, self.a[a2] + self.a[a1]])
            L = max([0, self.a[a2] + self.a[a1] - self.C])

        print >> sys.stderr, time.ctime(), "DEBUG, a2_new_value_unc:", a2_new_value, "H:", H, "L:", L
        
        if H < a2_new_value:
            return H
        elif L > a2_new_value:
            return L
        else:
            return a2_new_value

    def update_b(self, a1, a2, a1_new_value, a2_new_value):
        b1_new_value = -self.E[a1] - self.y[a1] * self.kernel(a1, a1) * (a1_new_value - self.a[a1]) - self.y[a2] * self.kernel(a2, a1) * (a2_new_value - self.a[a2]) + self.b
        b2_new_value = -self.E[a2] - self.y[a1] * self.kernel(a1, a2) * (a1_new_value - self.a[a1]) - self.y[a2] * self.kernel(a2, a2) * (a2_new_value - self.a[a2]) + self.b

        print >> sys.stderr, time.ctime(), "DEBUG, b1_new_value:", b1_new_value, "b2_new_value:", b2_new_value, "a1_index:", a1, "a2_index:", a2, "a[a1]:", self.a[a1], "a[a2]:", self.a[a2], "C:", self.C, "b:", self.b
        print >> sys.stderr, time.ctime(), "DEBUG, E[a1]", self.E[a1], "E[a2]", self.E[a2], "y[a1]", self.y[a1], "y[a2]", self.y[a2]
        print >> sys.stderr, time.ctime(), "DEBUG, K11:", self.kernel(a1, a1), "K12(k21):", self.kernel(a2, a1), "K22:", self.kernel(a2, a2)
        
        if 0 < self.a[a1] < self.C:
            self.b = b1_new_value
        elif 0 < self.a[a2] < self.C:
            self.b = b2_new_value
        else:
            self.b = (b1_new_value + b2_new_value) / 2.0 

        #check for update b
        if 0 < a1_new_value < self.C and 0 < a2_new_value < self.C:
            if not Tool.float_equal(b1_new_value, b2_new_value):
                print >> sys.stderr, time.ctime(), "WARNING, b1_new_value != b2_new_value when both are support_vectors, a1_new_value:", a1_new_value, "a2_new_value:", a2_new_value, "b1_new_value:", b1_new_value, "b2_new_value:", b2_new_value

    def train_given_a1_a2(self, a1, a2):
        # get new a1 and a2
        square_sum_of_kernel_a1_a2 = self.square_sum_of_kernel(a1, a2)
        if Tool.float_equal(square_sum_of_kernel_a1_a2, 0):
            print >> sys.stderr, time.ctime(), "WARNING, square_sum_of_kernel_a1_a2 near to 0, value:", square_sum_of_kernel_a1_a2
            square_sum_of_kernel_a1_a2 = 1e-6  
        a2_new_value_unc = self.a[a2] + self.y[a2] * (self.E[a1] - self.E[a2]) / square_sum_of_kernel_a1_a2
        
        # customized new a2
        a2_new_value = self.customized_new_a(a1, a2, a2_new_value_unc)
        a1_new_value = self.a[a1] + self.y[a1] * self.y[a2] * (self.a[a2] - a2_new_value)
        print >> sys.stderr, time.ctime(), "DEBUG, a2_new_value_unc:", a2_new_value_unc, "a2_new_value:", a2_new_value, "a1_new_value:", a1_new_value
    
        # update b
        self.update_b(a1, a2, a1_new_value, a2_new_value)
        
        # update a1 and a2
        self.a[a1] = a1_new_value
        self.a[a2] = a2_new_value
        print >> sys.stderr, time.ctime(), "DEBUG, a1_new_value:", a1_new_value, "a2_new_value:", a2_new_value, "b_new:", self.b
        
        # update support_vectors
        self.update_support_vectors()

        # update Ea1, Ea2
        support_vectors_list = self.get_support_vectors_list()
        self.E[a1] = self.b - self.y[a1]
        for i in support_vectors_list:
            self.E[a1] += self.y[i] * self.a[i] * self.kernel(i, a1)

        self.E[a2] = self.b - self.y[a2]
        for i in support_vectors_list:
            self.E[a2] += self.y[i] * self.a[i] * self.kernel(i, a2)
        print >> sys.stderr, time.ctime(), "DEBUG, update E, E[a1]", self.E[a1], "E[a2]", self.E[a2]
        print >> sys.stderr, time.ctime(), "DEBUG, now E:"
        for i in range(self.N):
            print >> sys.stderr, self.E[i],
        print >> sys.stderr, ""
    
        # update 
        if 0 == self.trained_sample % self.update_score_batch and 0 != self.trained_sample:
            self.update_score_on_traindata()
            self.get_accurcay()
        self.trained_sample += 1
            
    def train_given_a1(self, a1):
        # choose a2 by maximise |E1 - E2| (just for speed training)
        if self.E[a1] < 0:
            a2 = self.E.index(max(self.E))
            print >> sys.stderr, time.ctime(), "DEBUG, a2:", a2, "by self.E[a1] < 0"
        else:
            a2 = self.E.index(min(self.E))
            print >> sys.stderr, time.ctime(), "DEBUG, a2:", a2, "by self.E[a1] >= 0"

        self.train_given_a1_a2(a1 ,a2)

    def train(self):
        self.update_support_vectors()
        self.update_score_on_traindata() # watse time!!!

        for i in range(self.N):
            while self.SV_is_updated:
                self.SV_is_updated = False
                support_vectors_list = self.get_support_vectors_list()
                for j in support_vectors_list:
                    if not self.is_satisfy_KKT(j):
                        print >> sys.stderr, time.ctime(), "DEBUG, a1:", j, "from support_vectors_list"
                        self.train_given_a1(j)
                
            support_vectors_dict = self.get_support_vectors_dict()
            if i in support_vectors_dict:
                continue

            if not self.is_satisfy_KKT(i):
                print >> sys.stderr, time.ctime(), "DEBUG, a1:", i, "out of support_vectors_list"
                self.train_given_a1(i)
                    
class Data:
    def __init__(self, filename):
        self.data_features = []
        self.data_labels = []
        self.filename = filename
        self.feature_dim = 0
        self.sample_num = 0

        self.features_split = "\t" # 1. split among features 2. split between label and data
        self.id_and_features_split = ":"

    def read_data(self):
        f = open(self.filename)
        print >> sys.stderr, time.ctime(), "TARCE, begin read_data, filename:", self.filename

        self.feature_dim = -1
        count = 0
        for line in f:
            line = line.strip()
            slots = line.split(self.features_split)
            len_slots = len(slots)
            if len_slots <= 1:
                print >> sys.stderr, time.ctime(), "ERROR, len_slots <= 1 in read_data"
                
            if "1" == slots[0]:
                label = 1
            else:
                label = -1
            self.data_labels.append(label)
    
            continuous_id = 0
            sample_features = []
            for term in slots[1:]:
                id, feature = term.split(self.id_and_features_split) # input id should start from 0 TODO: add check
                id = int(id)
                feature = float(feature)

                while id - 1 > continuous_id: # deal with sparse input
                    sample_features.append(0.0)
                    continuous_id += 1

                sample_features.append(feature)
                continuous_id += 1

            # check
            if -1 == self.feature_dim:
                self.feature_dim = len(sample_features) # init check
            elif self.feature_dim != len(sample_features):
                print >> sys.stderr, time.ctime(), "WARNING, in line:", count, "self.feature_dim != len(sample_features)", self.feature_dim, "!=", len(sample_features), " continue"
                continue

            self.data_features.append(sample_features)
            count += 1

        self.sample_num = len(self.data_labels)
        print >> sys.stderr, time.ctime(), "INFO, read_data done, feature_dim:", self.feature_dim, "sample_num:", self.sample_num

        f.close()

    def get_data(self):
        return self.data_features, self.data_labels

    def get_config_auto_from_data(self):
        return self.feature_dim, self.sample_num
        
def init(config, model, train_data):
    train_data.read_data()

    config.feature_dim, config.train_sample_num = train_data.get_config_auto_from_data()

    model.set_config_and_init(config)

    model.x, model.y = train_data.get_data()

    print >> sys.stderr, time.ctime(), "TRACE, init done"

def train(model):
    model.train_all_epoch()

def main(argv):
    if len(argv) != 2:
        print >> sys.stderr, time.ctime(), "Usage:", argv[0], "train_file"
    train_file = argv[1]

    print >> sys.stderr, time.ctime(), "TRACE, begin"

    config = Config()
    model = Model()
    train_data = Data(train_file)
#    Data predict_data(predict_file)
    
    init(config, model, train_data)

    train(model)

    print >> sys.stderr, time.ctime(), "TRACE, totally done"
    
if "__main__" == __name__:
    main(sys.argv)
