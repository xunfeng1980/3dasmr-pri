apt-get  install zlib1g-dev libcunit1-dev libcunit1-dev git build-essential cmake nodejs  -y
cd ~/libmysofa/
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j8 all
make install

gcc -o export_hrtf main.c -lmysofa   -lm

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
./export_hrtf  Kemar_HRTF_sofa.sofa