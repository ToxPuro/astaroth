import numpy as np
NGHOST = int(3)
NPOINTS = int(NGHOST*2+1)
WIDTH = float(NGHOST)-1.0
def base_gaussian():
    res = []
    for i in [-3,-2,-1,0,1,2,3]:
        res.append(np.exp(-float(i*i)/(2.0*WIDTH)))
    return np.array(res)
def main():
    arr = base_gaussian()
    coeffs = np.zeros((NPOINTS,NPOINTS,NPOINTS))
    for i in [-3,-2,-1,0,1,2,3]:
        for j in [-3,-2,-1,0,1,2,3]:
            for k in [-3,-2,-1,0,1,2,3]:
                coeffs[i+NGHOST][j+NGHOST][k+NGHOST] = arr[i+NGHOST]*arr[j+NGHOST]*arr[k+NGHOST]
    coeff_sum  = np.sum(coeffs)
    for i in [-3,-2,-1,0,1,2,3]:
        for j in [-3,-2,-1,0,1,2,3]:
            for k in [-3,-2,-1,0,1,2,3]:
                coeffs[i+NGHOST][j+NGHOST][k+NGHOST] = arr[i+NGHOST]*arr[j+NGHOST]*arr[k+NGHOST]/coeff_sum
                print(f"\t  [{i}][{j}][{k}] = {coeffs[i+NGHOST][j+NGHOST][k+NGHOST]},")
if __name__ == '__main__':
        main()
