#!/bin/python3
import numpy as np
import sys
import matplotlib.pyplot as plt

# Data formats
# data-format.csv: specifies precision and data dims
# <field>-data.out: a binary file storing the data within <field>

if len(sys.argv) <= 2:
    print('Usage: ./analysis.py <header file> <data file>')
    exit(-1)

headerpath = sys.argv[1]
datapath = sys.argv[2]


header = np.genfromtxt(headerpath, dtype=int, delimiter=',', skip_header=1)
use_double, mx, my, mz = header[0], header[1], header[2], header[3]

data = np.fromfile(datapath, dtype = np.double if use_double else np.float)
data = np.reshape(data, (mx, my, mz))

slice = data[:, :, int(mz/2)]

#print(header)
#print(data)
#print(slice)

plt.pcolormesh(slice)
plt.axis('scaled')
plt.savefig(datapath + '.png')
#plt.show()