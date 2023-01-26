import random
a = [1,2,3]
b = [1,4,5]
a, b = set(a), set(b)
a |= b
print(a)
