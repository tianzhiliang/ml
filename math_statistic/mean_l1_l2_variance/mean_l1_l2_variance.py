import sys
import math

def mean(nums):
    return sum(nums) * 1.0 / len(nums)

def l1(nums):
    sum = 0
    for num in nums:
        sum += math.fabs(num)
    return sum
    
def l2(nums):
    sum = 0
    for num in nums:
        sum += num ** 2
    return math.sqrt(sum)
    
def variance(nums, mean):
    sum = 0
    for num in nums:
        sum += (num - mean) ** 2
    return math.sqrt(sum)
    
nums = []
for line in sys.stdin:
    line = line.strip()
    nums.append(float(line))

mean_value = mean(nums)
print "mean:", mean_value
print "l1:", l1(nums)
print "l2:", l2(nums)
print "variance:", variance(nums, mean_value)
