def merge(intervals):
    intervals.sort(key=lambda x: x[0])   # sort by start time
    output = [intervals[0]]

    for start, end in intervals[1:]:
        lastEnd = output[-1][1]
        # overlap → merge
        if start <= lastEnd:
            output[-1][1] = max(lastEnd, end)
        else:
            output.append([start, end])

    return output

print(merge([[1,3],[2,6],[8,10],[15,18]]))