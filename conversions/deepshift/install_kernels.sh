#/usr/bin/sh
cd ./kernels/cpu
python setup.py install
cd -

cd ./kernels/cuda
python setup.py install
cd -
