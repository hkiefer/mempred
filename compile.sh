rm ./mempred/ckernel.cpython*
rm ./build/* -rf
rm ./mempred/__pycache__/*
python3 setup-ckernel.py build_ext --inplace
