#!/usr/bin/python3
#two sum code in python o(n)
def two_sum(nums, target): 
    num_map = {}
    for i, num in enumerate(nums):
        print(i, num)
        print(num_map)
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i], [complement, num]
        num_map[num] = i
        print(num_map)
    



print(two_sum([2, 2, 7, 5, 4], 9))


def two_sum_all_combos(nums, target): 
    num_map = {}
    out_list_ind = []
    out_list_num = []
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            out_list_ind.append([num_map[complement], i])
            out_list_num.append([complement, num])
        num_map[num] = i
        
    return out_list_ind, out_list_num

print(two_sum_all_combos([1, 1, 2, 3],4))