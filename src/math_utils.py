def euclidean_dist(a, b):
    cols = len(a)
    #print(a)
    #print(b)
    res  = [(a[idx]-b[idx])**2 \
        for idx in range(cols)]
    res = sum(res)**1/2
    #print("Euclidean distance: ", res)
    return res

def manhattan_dist(a, b):
    cols = len(a)
    res  = sum([abs(a[idx]-b[idx]) \
        for idx in range(cols)])
    #print("Manhattan distance: ", res)
    return res

def supreme_dist(a, b):
    cols = len(a)
    res  = [abs(a[idx]-b[idx]) \
        for idx in range(cols)]
    max_value = max(res)
    #print("Supreme distance: ", res)
    return res