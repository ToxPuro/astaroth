
rm justtest.so

python setup.py build
ln -s build/lib.linux-x86_64-3.7/justtest.cpython-37m-x86_64-linux-gnu.so justtest.so 
