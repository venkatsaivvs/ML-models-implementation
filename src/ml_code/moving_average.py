

def moving_average(arr, window_size):
    result = []
    for i in range(len(arr) - window_size + 1):
        window_sum = sum(arr[i:i + window_size])
        result.append(window_sum / window_size)
    return result

# Example
data = [1, 2, 3, 4, 5, 6]
print(moving_average(data, 3))
# Output: [2.0, 3.0, 4.0, 5.0]


from collections import deque

class MovingAverage:
    def __init__(self, window_size):
        self.queue = deque()
        self.window_size = window_size
        self.window_sum = 0

    def next(self, val):

        self.queue.append(val)
        self.window_sum+=val


        if len(self.queue) > self.window_size:
            self.window_sum -= self.queue.popleft()

        return self.window_sum/len(self.queue)

m = MovingAverage(3)
print(m.next(1))   # 1.0
print(m.next(10))  # 5.5
print(m.next(3))   # 4.6667
print(m.next(5))   # 6.0





