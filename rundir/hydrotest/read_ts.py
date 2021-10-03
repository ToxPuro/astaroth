#!/usr/bin/python3
import numpy as np
import sys

file = sys.argv[1]

with open(file, "rb") as f:
    h = len(b'AcRealSize=8;Endianness=LE;')
    t = f.read()
    ts = np.frombuffer(t[h:])

print("length: ", ts.shape)
print(ts)
for i in range(len(ts)):
    b = 8*i
    c = t[h+b:h+b+8]
    print(f"{i}: {c.hex()} --> {np.frombuffer(c)[0]}")
