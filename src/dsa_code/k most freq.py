

def k_most_freq(nums, k):
    count_map = {}
    for n in nums:
        if n in count_map:
            count_map[n] += 1
        else:
            count_map[n] = 1
    
    sorted_count_map = sorted(count_map.items(), key=lambda x: x[1], reverse=True)
    return sorted_count_map[:k]

print(k_most_freq([1,1,1,2,2,3], 2))