
import numpy as np

print("hello, lets test this!\n")
a = np.random.random(26)+0.5
for i in [1,4,3,22,5]:
    a[i] = 0

filled = np.nonzero (a)
print(f'filled i = {filled}')

v = np.int_(a)
print(f'trunced ints: {v}')