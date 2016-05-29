import sys
import os
import math

SLOT_SPLIT = "\t"

def element_mul(a, b):
    length = len(a)
    if 0 == length:
        print >> sys.stderr, "ERROR, 0 == length"
        return 0
    if length != len(b):
        print >> sys.stderr, "ERROR, len(a):", length, " != len(b):", len(b)
        return 0

    ab = []
    for i in range(length):
        ab.append(a[i] * b[i])
    return ab

def vector_sum(list):
    return sum(list)

def pearson(a, b):
    length = len(a)
    if 0 == length:
        print >> sys.stderr, "ERROR, 0 == length"
        return 0
    if length != len(b):
        print >> sys.stderr, "ERROR, len(a):", length, " != len(b):", len(b)
        return 0

    length = length * 1.0
    
    sum_a = vector_sum(a)
    sum_b = vector_sum(b)
    ele_mul_a_b = element_mul(a, b)
    ele_mul_a_a = element_mul(a, a)
    ele_mul_b_b = element_mul(b, b)
    
    E_ab = vector_sum(ele_mul_a_b)
    E_aa = vector_sum(ele_mul_a_a)
    E_bb = vector_sum(ele_mul_b_b)
    E_a_E_b = sum_a * sum_b / length
    E_a_E_a = sum_a * sum_a / length
    E_b_E_b = sum_b * sum_b / length

    r1 = (E_ab - E_a_E_b)
    r2 = (E_aa - E_a_E_a) * (E_bb - E_b_E_b)
    r2 = math.sqrt(r2)

    r = r1 / r2
    return r

def main(argv):
    if 2 == len(argv):
        if "h" in argv[1] or "help" in argv[1]:
            print >> sys.stderr, "input streaming has two columns, split by tab. eg: data_a_i <tab> data_b_i"
            return True
        
    a = []
    b = []
    for line in sys.stdin:
        line = line.strip()
        slots = line.split(SLOT_SPLIT)
        if 2 != len(slots):
            print >> sys.stderr, "WARNING, 2 != len(slots), line:", line, "len:", len(slots)
            continue
        a.append(slots[0])
        b.append(slots[1])
    a = map(float, a)
    b = map(float, b)
    print >> sys.stderr, "input file size:", len(a)

    r = pearson(a, b)

    print "Pearson:", r
    return True

if "__main__" == __name__:
    main(sys.argv)
