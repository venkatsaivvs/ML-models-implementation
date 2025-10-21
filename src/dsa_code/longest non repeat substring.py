#!/usr/bin/python3
#longest non repeat substring code in python o(n)
def longest_non_repeat(s):
    char_set = set()
    max_length = 0
    l = 0
    for r in range(len(s)):
        while s[r] in char_set:
            char_set.remove(s[l])
            l += 1
        char_set.add(s[r])
        print(char_set)
        max_length = max(max_length, r - l + 1)
    return max_length

print(longest_non_repeat("abcabcbb"))