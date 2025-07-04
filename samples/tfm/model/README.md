# Reproducing the results



## Generating model solutions

Laplace SOCA Roberts
1. Set `hostdefine LHYDRO (0)`
```bash
CASE=laplace-soca-roberts
CONFIG=~/astaroth/samples/tfm/cases/laplace-soca.ini
MODELDIR=~/astaroth/samples/tfm/model/${CASE}
```

Laplace Non-SOCA Turbulence
1. Set `hostdefine LHYDRO (1)`
```bash
CASE=laplace-nonsoca-turbulence
CONFIG=~/astaroth/samples/tfm/cases/laplace-nonsoca.ini
MODELDIR=~/astaroth/samples/tfm/model/${CASE}
```


```bash
# In build directory (use with care, calls `rm`)
mkdir -p ${MODELDIR} &&\
rm -rf simulation_state.txt tfm-mpi && \
cmake ~/astaroth/ --preset lumi-tfm && \
cmake --build . --parallel && \
rm -rf *.snapshot *.profile *.mesh simulation_state.txt && \
 $SRUNMPI1 ./tfm-mpi --config ${CONFIG} && \
rm -rf output && ~/astaroth/samples/tfm-mpi/visualize.py --inputs *2500* && \
~/astaroth/samples/tfm-mpi/view-timeseries.py && \
md5sum *.snapshot > ${MODELDIR}/snapshots.txt && \
cp timeseries.csv ${MODELDIR}/ &&\
mv output ${MODELDIR}/ &&\
cp ${CONFIG} ${MODELDIR}/ &&\
git rev-parse HEAD > ${MODELDIR}/head.txt &&\
git --no-pager diff > ${MODELDIR}/diff.txt
```


## PC model SOCA Roberts sanity check

uurms, max: 1.000000e+00,1.407404e+00
bbrms, max: 5.773465e+00,9.344131e+00

```
# Astaroth a21
aatest:
0x -9.292349815368652 9.292349815368652
0y 0.0 0.0
0z -6.5706868171691895 6.5706868171691895

# Astaroth a11
1x 0.0 0.0
1y -9.292349815368652 9.292349815368652
1z -6.5706868171691895 6.5706868171691895

# Astaroth a12
2x 0.0 0.0
2y -9.292349815368652 9.292349815368652
2z -6.5706868171691895 6.5706868171691895

# Astaroth a22
3x -9.292349815368652 9.292349815368652
3y 0.0 0.0
3z -6.5706868171691895 6.5706868171691895
```

TF_a11_x,2500,8.160662e+01,3.262428e-02,-4.259346e-17,1.527871e-17,4.259346e-17,-2.818777e-34
TF_a11_y,2500,8.160662e+01,3.262428e-02,-9.292505e+00,3.333318e+00,9.292505e+00,1.052354e-16
TF_a11_z,2500,8.160662e+01,3.262428e-02,-6.570793e+00,2.357011e+00,6.570793e+00,2.075951e-17
TF_a12_x,2500,8.160662e+01,3.262428e-02,-4.259346e-17,1.527871e-17,4.259346e-17,7.297175e-35
TF_a12_y,2500,8.160662e+01,3.262428e-02,-9.292505e+00,3.333318e+00,9.292505e+00,5.426093e-18
TF_a12_z,2500,8.160662e+01,3.262428e-02,-6.570793e+00,2.357011e+00,6.570793e+00,-5.839022e-18
TF_a21_x,2500,8.160662e+01,3.262428e-02,-9.292505e+00,3.333318e+00,9.292505e+00,-7.878846e-17
TF_a21_y,2500,8.160662e+01,3.262428e-02,-8.955183e-17,3.196936e-17,8.955183e-17,7.095578e-35
TF_a21_z,2500,8.160662e+01,3.262428e-02,-6.570793e+00,2.357011e+00,6.570793e+00,-2.002089e-17
TF_a22_x,2500,8.160662e+01,3.262428e-02,-9.292505e+00,3.333318e+00,9.292505e+00,-3.264719e-17
TF_a22_y,2500,8.160662e+01,3.262428e-02,-8.955183e-17,3.196936e-17,8.955183e-17,-4.764910e-34
TF_a22_z,2500,8.160662e+01,3.262428e-02,-6.570793e+00,2.357011e+00,6.570793e+00,4.579865e-17

## Verifying


### build
`cmake ~/astaroth/ --preset lumi-tfm && cmake --build . --parallel`

### laplace nonsoca turbulence:
`$SRUNMPI64 ./tfm-mpi --config ~/astaroth/samples/tfm/cases/laplace-nonsoca.ini --benchmark 1`

### laplace soca roberts:
`$SRUNMPI64 ./tfm-mpi --config ~/astaroth/samples/tfm/cases/laplace-soca.ini --benchmark 1`

### Compare with ´sample/tfm-mpi/verify-allclose.py´