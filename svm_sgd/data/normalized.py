import os
import sys

def norm(slots):
    label = slots[0]
    result = []
    for term in slots[1:]:
        id, feature = term.split(":")
        if not "0" == feature: # for speed
            feature = float(feature) / 256
            term_res = id + ":" + str(feature)
            result.append(term_res)
        else:
            result.append(term)
    print label + "\t" + "\t".join(result)

for line in sys.stdin:
    slots = line.strip().split("\t")
    norm(slots)
