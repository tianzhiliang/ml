import sys
import os
import math

def read_data(file):
    f = open(file, "r")
    datas = []
    for l in f:
        data = l.strip().split()
        data = map(float, data)
        datas.append(data)
    f.close()
    return datas

def load_data_and_init(datafile, initfile):
    data = read_data(datafile)
    init_cluster = read_data(initfile)
    return data, init_cluster

def euc_dis(a, b):
    sum = 0
    for aa, bb in zip(a, b):
        sum += (aa - bb) ** 2
    return math.sqrt(sum)

def near_search(query, cands):
    min_index = -1
    min_value = 100000
    for i, candidate in enumerate(cands):
        value = euc_dis(query, candidate)
        if value < min_value:
            min_value = value
            min_index = i
        #print("i, value, min_index:", i, value, min_index)
    #print("min_index:", min_index)
    return min_index

def near_search_with_debug(dataID, query, cands):
    min_index = -1
    min_value = 100000
    for i, candidate in enumerate(cands):
        value = euc_dis(query, candidate)
        if value < min_value:
            min_value = value
            min_index = i
        print("dataID: " + str(dataID) + " cluID: " + str(i) + " Dist: " + str(value))
    print("dataID: " + str(dataID) + " NearestCluster: " + str(min_index))
    return min_index

def map2list(dict): # the key of map is from 0 to len(dict.keys())
    lenl = len(dict.keys())
    #print("lenl:", lenl, " dict.keys(): ", dict.keys())
    data = []
    for i in range(lenl):
        data.append(dict[i])
    return data

def assign_category(data, cluster):
    data2cluster = []
    cluster2data = {}

    print("data:", data)
    for i, d in enumerate(data):
        data2cluster.append(near_search_with_debug(i, d, cluster))

    #print("data2cluster:", data2cluster)
    for i, clu in enumerate(data2cluster):
        if not clu in cluster2data:
            cluster2data[clu] = [i]
        else:
            cluster2data[clu].append(i)

    #print("cluster2data:", cluster2data)
    print("cluster2data:", map2list(cluster2data))
    return data2cluster, map2list(cluster2data)

def get_mean_of_data(data):
    dim = len(data[0])
    size = len(data)

    mean = [0 for i in range(dim)] 

    for d in data:
        for i, value in enumerate(d):
            mean[i] += value

    #print("mean:", mean)
    for i, m in enumerate(mean):
        mean[i] = m / size
        #print("m, size, mean:", m, size, mean[i])

    return mean

def get_subset_from_data_by_indexs(data, indexs):
    subset = []
    for index in indexs:
        subset.append(data[index])
    return subset

def update_cluster(data, data2cluster, cluster2data):
    cluster = []
    for i, clu in enumerate(cluster2data):
        data_under_clu = get_subset_from_data_by_indexs(data, clu)
        print("data under cluster " + str(i) + "-th: ", data_under_clu)
        mean = get_mean_of_data(data_under_clu)
        print("mean of cluster " + str(i) + "-th: ", mean)
        cluster.append(mean)
    print("cluster:", cluster)
    return cluster

def print_cluster(cluster):
    for i, clu in enumerate(cluster):
        print("Cluster: ", i, " Mean: ", clu)

def print_cluster2data(cluster2data):
    for i in range(len(cluster2data)):
        print("Cluster: ", i, " Data: ", cluster2data[i])

def KM(data, init_cluster, iterator):
    cluster = init_cluster
    for i in range(iterator):
        print("Iterator:", i)
        data2cluster, cluster2data = assign_category(data, cluster)
        cluster = update_cluster(data, data2cluster, cluster2data)
        print("")
    print_cluster(cluster)
    print_cluster2data(cluster2data)

def main():
    # Branch 1: load init and K from file 
    data, init_cluster = load_data_and_init(sys.argv[1], sys.argv[2])
    # Branch 2: init cluster according to K
    iterator = int(sys.argv[3])
    KM(data, init_cluster, iterator)

main()
