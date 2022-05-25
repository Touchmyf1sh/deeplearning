import numpy as np

dt = np.dtype([('xuexi', 'i4'), ('123', np.int)])
a = np.array([[(1, 2), (2, 2), (3, 3)]], dtype=dt)
for i in a.flat:
    for q in i:
        print(q)