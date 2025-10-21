
from typing import KeysView


distances = {
    0: 10,
    1: 12,
    2: 11
}
#sort entire dictionary by values and get dictionary of sorted indices and values
sorted_dict = dict(sorted(distances.items(), key=lambda item: item[1]))
print(sorted_dict)
neighbor_indices = list(sorted_dict.keys())[:3]
neighbor_distances = [sorted_dict[key] for key in neighbor_indices]
print(neighbor_distances)

#just like in list can i get first n elements inluding Keys of dictionary?

from collections import Counter
counter = Counter([1,1,1,2,2,2,2])
print(counter.most_common(1)[0])