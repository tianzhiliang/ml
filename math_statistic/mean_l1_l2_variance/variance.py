import sys
import math

def variance(nums):
    mean = sum(nums) / len(nums)
    res = 0
    for num in nums:
        res += (num - mean) ** 2
    return math.sqrt(res)

nums = []
for line in sys.stdin:
    line = line.strip()
    nums.append(line)
    
nums = map(float, nums)
print variance(nums)
