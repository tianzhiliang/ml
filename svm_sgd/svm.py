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

    @staticmethod
    def vec_mul_num(vec_src, num, vec_dest):
        length = len(vec_src)
        for i in range(length):
            vec_dest[i] = vec_src[i] * num

    @staticmethod
    def vec_mul_num_add(vec_src, num, vec_dest):
        length = len(vec_src)
        for i in range(length):
            vec_dest[i] += vec_src[i] * num

class Config:
    def __init__(self):
#    def __init__(self, relaxation_factor, w_random_range, b_random_range, a_random_range, total_epoch, kernel_type):
        self.train_sample_num = 0
        self.predict_sample_num = 0
        self.feature_dim = 0
        self.relaxation_factor = 0.5
        self.w_random_range = 0.5 # (-w_random_range, w_random_range)
        self.b_random_range = 0.5 # b_random_range 
        self.a_random_range = 0
        self.total_epoch = 30
        self.kernel_type = 0
        self.update_score_batch = 32
        self.hingloss_margin = 1
        self.batch_size = 512
        self._lambda = 1
        self.save_model_samples = 10240


class Model:
    def __init__(self):
        # model parameter, hyper-parameter, important variables
        self.a = [] # Lanrange multiplier
        self.E = [] # loss
        self.b = [] # bias parameter
        self.C = 0 # relaxation_factor
        self.W = [] # weight parameter
        self.W2 = [] # weight parameter
        self.config = None
        self.dim = 0
        self.N = 0
        self.total_epoch = 0
        self.kernel_type = 0
        self.update_score_batch = 0
        self.hingloss_margin = 0
        self.batch_size = 0
        self._lambda = 0.0
        self.save_model_samples = 0

        # data
        self.x = [] # samples' input features, a matrix
        self.y = [] # samples' label (ground-turth)

        # temp variable
        self.g = [] # score of each sample, g(x) = \sigma (w_i * x_i) _ b
        self.support_vector_dict = {}

        # sign and stat
        self.SV_is_updated = True # support vector list have been updated or not, since the lastest read
        self.traind_epoch = 0
        self.batch_right_count = 0
        self.total_right_count = 0
        self.total_trained_sample = 0
        self.save_iter = 0
        self.learing_rate = 0 # is not a hyper-parameter in pegasos-svm
        self.total_batch_iter = 0
        

    def set_config_and_init(self, config):
        self.config = config
        self.b = (random.random() - 1) * 2 * config.b_random_range
        #self.a = [(random.random() - 1) * 2 * a_random_range for i in range(config.train_sample_num)] # TODO ??? 
        self.a = [0 for i in range(config.train_sample_num)]
        self.E = [0 for i in range(config.train_sample_num)]
        self.W = [(random.random() - 1) * 2 * config.w_random_range for i in range(config.feature_dim)]
        self.W2 = [0 for i in range(config.feature_dim)]
        self.g = [0 for i in range(config.train_sample_num)]
        self.C = config.relaxation_factor
        self.dim = config.feature_dim
        self.N = config.train_sample_num
        self.total_epoch = config.total_epoch
        self.kernel_type = config.kernel_type
        self.update_score_batch = config.update_score_batch
        self.hingloss_margin = config.hingloss_margin
        self.batch_size = config.batch_size
        self._lambda = config._lambda
        self.save_model_samples = config.save_model_samples
        print >> sys.stderr, time.ctime(), "INFO, set_config_and_init done, train_sample_num:", self.N, "feature_dim", self.dim, "relaxation_factor:", self.C, "total_epoch:", self.total_epoch, "kernel_type:", self.kernel_type, "b_random_range:", config.b_random_range, "w_random_range:", config.w_random_range, "lambda:", self._lambda, "batch_size:", self.batch_size, "hingloss_margin:", self.hingloss_margin, "save_model_samples:", self.save_model_samples

    def train_all_epoch(self):
        print >> sys.stderr, time.ctime(), "TRACE, begin train_all_epoch"

        for i in range(self.total_epoch):
            print >> sys.stderr, time.ctime(), "TRACE, begin train epoch:", i, ". Total epoch:", self.total_epoch
            self.train_one_epoch()
            self.traind_epoch += 1

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

    def score_on_traindata_for_debug(self):
        right = 0
        for i in range(self.N):
            score = Tool.dot_product(self.W, self.x[i])
            predict_value = score * self.y[i]
            if predict_value >= 0:
                right += 1
            print >> sys.stderr, time.ctime(), "DEBUG, All traindata, offset:", i, "y:", self.y[i], "dot(w,x):", score, "predict_value:", predict_value, "right_count:", right
        print >> sys.stderr, time.ctime(), "INFO, All traindata now accuracy:", right * 1.0 / self.N
    
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
        print >> sys.stderr, time.ctime(), "INFO, epoch", self.traind_epoch, "samples in this epoch:", self.total_trained_sample, "accuracy:", acc

    def save_model(self):
        touch_save = False
        if os.path.exists("touch_save_model"):
            touch_save = True
        if self.save_iter + 1 == self.total_trained_sample / self.save_model_samples or touch_save:
            if not touch_save:
                self.save_iter = self.total_trained_sample / self.save_model_samples
                filename = "model.iter" + str(self.save_iter)
            else:
                filename = "hand.model"
            
            f = open(filename, "w")
            f.write(str(self.dim) + "\n")
            for w in self.W:
                f.write(str(w) + "\n")
            f.close()

            if touch_save:
                os.remove("touch_save_model")

    def training_accuracy(self, t):
        batch_size = min([self.batch_size, self.N - self.batch_size * t])
        accuracy = self.batch_right_count * 1.0 / batch_size
        self.total_right_count += self.batch_right_count
        self.total_trained_sample += batch_size
        total_accuracy = self.total_right_count * 1.0 / self.total_trained_sample 
        self.batch_right_count = 0

        print >> sys.stderr, time.ctime(), "INFO, epoch:", self.traind_epoch, " batch:", t, "total:", self.total_trained_sample, "batch_accuracy:", accuracy, "total_accuracy:", total_accuracy

    def train_one_epoch(self):
        batch_iter = self.N / self.batch_size
        if 0 != self.N % self.batch_size:
            batch_iter += 1

        for t in range(batch_iter):
            # step 0: ff
            batch_size = min([self.batch_size, self.N - self.batch_size * t])
            gradient_by_hingeloss = [[0 for i1 in range(batch_size)] for i2 in range(self.dim)]
#            print >> sys.stderr, time.ctime(), "DEBUG, N:", self.N, "t:", t, "self.batch_size:", self.batch_size, "real_batch_size:", batch_size
            for j in range(batch_size):
                offset = t * batch_size + j
                predict_value = self.y[offset] * Tool.dot_product(self.W, self.x[offset])
                if predict_value >= 0:
                    self.batch_right_count += 1
                if predict_value < self.hingloss_margin: # not satisfy margin -> need update -> hinge loss value is non-zero
                    for k in range(self.dim):
                        gradient_by_hingeloss[k][j] = self.x[j][k] * self.y[j] # gredient of weight parameter
                print >> sys.stderr, time.ctime(), "DEBUG, offset:", offset, "y:", self.y[offset], "dot(w,x):", Tool.dot_product(self.W, self.x[offset]), "predict_value:", predict_value, "batch_right_count:", self.batch_right_count
            
            # step 1: get W2
            self.learing_rate = 1.0 / ((self.total_batch_iter + 1) * self._lambda) # time is start from 1, avoiding "float division by zero"

            # gradient_by_hingeloss
            for k in range(self.dim):
                self.W2[k] = sum(gradient_by_hingeloss[k]) # sum of batch samples' gradient at k-th dim

            # learing_rate on gradient_by_hingeloss
            Tool.vec_mul_num(self.W2, self.learing_rate / batch_size, self.W2)

            # normlized term
            Tool.vec_mul_num_add(self.W, 1 - 1.0/(self.total_batch_iter + 1), self.W2)

            # step 2: update W
            # normlized (necessary with hingeloss)
            #W2_norm_term = Tool.dot_product(self.W2, self.W2)  # is it need sqrt ???
            W2_norm_term = math.sqrt(Tool.dot_product(self.W2, self.W2))  # is it need sqrt ???
            if Tool.float_equal(W2_norm_term, 0):
                W2_norm_term = 1e-5
            norm_factor_unc = 1.0 / math.sqrt(self._lambda) / W2_norm_term
            norm_factor = min([1, norm_factor_unc])

            Tool.vec_mul_num(self.W2, norm_factor, self.W)
            print >> sys.stderr, time.ctime(), "DEBUG, W2_norm_term:", W2_norm_term, "lambda:", self._lambda, "lr:", self.learing_rate, "norm_factor_unc:", norm_factor_unc, "norm_factor:", norm_factor
            
            if 0 == self.total_trained_sample % 4096:
                self.score_on_traindata_for_debug()
            self.training_accuracy(t)
            self.save_model()
            self.total_batch_iter += 1

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
