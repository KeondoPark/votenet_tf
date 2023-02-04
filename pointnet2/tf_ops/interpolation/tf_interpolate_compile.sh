CUDA_ROOT=/usr/local/cuda-12.0

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


#${CUDA_ROOT}/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu ${TF_CFALGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 -shared -o tf_grouping_so.so tf_grouping.cpp tf_grouping_g.cu.o ${TF_CFLAGS[@]} -fPIC -L${CUDA_ROOT}/lib64 -lcudart ${TF_LFLAGS[@]} -I ${CUDA_ROOT}/include

#TF 2.5
# g++ -std=c++11 -shared -o tf_interpolate_so_server.so tf_interpolate.cpp ${TF_CFLAGS[@]} -fPIC ${TF_LFLAGS[@]}


${CUDA_ROOT}/bin/nvcc -std=c++11 -c -o tf_interpolate_server.cu.o tf_interpolate.cu \
    ${TF_CFALGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o tf_interpolate_so_server.so tf_interpolate.cpp tf_interpolate_server.cu.o \
    ${TF_CFLAGS[@]} -fPIC -L${CUDA_ROOT}/lib64 -lcudart ${TF_LFLAGS[@]} -I ${CUDA_ROOT}/include



#TF 2.8
# g++ -std=c++14 -shared -o tf_interpolate_so_server_latest.so tf_interpolate.cpp ${TF_CFLAGS[@]} -fPIC ${TF_LFLAGS[@]}
