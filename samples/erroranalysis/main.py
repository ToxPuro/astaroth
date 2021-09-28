import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys

nn = 128

if len(sys.argv) != 3:
    print("Usage: ./main.py <input> <output>")
    quit()

input = str(sys.argv[1])
output = str(sys.argv[2])

z = np.loadtxt(input)

#z = np.loadtxt('errors.out')
#z = np.random.rand(nn, nn)

x = np.linspace(0, nn, 1)
y = np.linspace(0, nn, 1)


plot = plt.imshow(z, interpolation='bilinear', vmax = 5)
plt.colorbar()


# Show
write_to_file = True
#write_to_file = False
if write_to_file:
    plt.savefig(output, bbox_inches='tight')
else:
    plt.show()

print("Success")