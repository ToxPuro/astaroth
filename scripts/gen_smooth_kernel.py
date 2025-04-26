import numpy as np
NGHOST = int(3)
NPOINTS = int(NGHOST*2+1)
WIDTH = float(NGHOST)-1.0
def gaussian_weights():
    res = []
    for i in [-3,-2,-1,0,1,2,3]:
        res.append(np.exp(-float(i*i)/(2.0*WIDTH)))
    return np.array(res)
def gen_stencil(weights):
    coeffs = np.zeros((NPOINTS,NPOINTS,NPOINTS))
    for i in [-3,-2,-1,0,1,2,3]:
        for j in [-3,-2,-1,0,1,2,3]:
            for k in [-3,-2,-1,0,1,2,3]:
                coeffs[i+NGHOST][j+NGHOST][k+NGHOST] = weights[i+NGHOST]*weights[j+NGHOST]*weights[k+NGHOST]
    coeff_sum  = np.sum(coeffs)
    for i in [-3,-2,-1,0,1,2,3]:
        for j in [-3,-2,-1,0,1,2,3]:
            for k in [-3,-2,-1,0,1,2,3]:
                coeffs[i+NGHOST][j+NGHOST][k+NGHOST] = weights[i+NGHOST]*weights[j+NGHOST]*weights[k+NGHOST]/coeff_sum
                print(f"\t  [{i}][{j}][{k}] = {coeffs[i+NGHOST][j+NGHOST][k+NGHOST]},")
def main():
    ##TP: for gaussian smoothing
    ##weights = base_gaussian()
    ##TP: for arbitrary smoothing
    weights = np.array([1.0, 9.0, 45.0, 70.0, 45.0, 9.0, 1.0])
    gen_stencil(weights)
if __name__ == '__main__':
        main()
