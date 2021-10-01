#/bin/bash
CUDA_ROOT=/usr/local/cuda-11.4
#TF_ROOT=/home/gundo0102/anaconda3/envs/votenet_tf/lib/python3.7/site-packages/tensorflow
#${CUDA_ROOT}/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/gundo0102/anaconda3/envs/votenet_tf/lib/python3.7/site-packages/tensorflow/include -I /usr/local/cuda-10.1/include -lcudart -L /usr/local/cuda-10.1/lib64/ -O2


# TF1.4
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L${TF_ROOT} -ltensorflow_framework -O2 


TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


#${CUDA_ROOT}/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu ${TF_CFALGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
${CUDA_ROOT}/bin/nvcc -std=c++11 -c -o tf_sampling_g_server.cu.o tf_sampling_g.cu ${TF_CFALGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o tf_sampling_so_server.so tf_sampling.cpp tf_sampling_g_server.cu.o ${TF_CFLAGS[@]} -fPIC -L${CUDA_ROOT}/lib64 -lcudart ${TF_LFLAGS[@]} -I ${CUDA_ROOT}/include
