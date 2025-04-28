import numpy as np
NGHOST = 3
NPOINTS = int(NGHOST*2+1)
WIDTH = float(NGHOST)-1.0
def gaussian_weights():
    res = []
    for i in range(-NGHOST,NGHOST+1):
        res.append(np.exp(-float(i*i)/(2.0*WIDTH)))
    return np.array(res)
def gen_stencil(weights):
    coeffs = np.zeros((NPOINTS,NPOINTS,NPOINTS))
    for i in range(-NGHOST,NGHOST+1):
        for j in range(-NGHOST,NGHOST+1):
            for k in range(-NGHOST,NGHOST+1):
                coeffs[i+NGHOST][j+NGHOST][k+NGHOST] = weights[i+NGHOST]*weights[j+NGHOST]*weights[k+NGHOST]
    coeff_sum  = np.sum(coeffs)
    for i in range(-NGHOST,NGHOST+1):
        for j in range(-NGHOST,NGHOST+1):
            for k in range(-NGHOST,NGHOST+1):
                coeffs[i+NGHOST][j+NGHOST][k+NGHOST] = weights[i+NGHOST]*weights[j+NGHOST]*weights[k+NGHOST]/coeff_sum
                print(f"\t  [{i}][{j}][{k}] = {coeffs[i+NGHOST][j+NGHOST][k+NGHOST]},")
                ##print(f"\t  [{i}][{j}][{k}] = 1,")
def main():
    ##TP: for gaussian smoothing
    ##weights = base_gaussian()
    ##TP: for arbitrary smoothing
    weights = np.array([1.0, 9.0, 45.0, 70.0, 45.0, 9.0, 1.0])
    #TP: for radius of 2 smoothing
    ##weights = np.array([45.0, 70.0, 45.0])
    gen_stencil(weights)
if __name__ == '__main__':
        main()
