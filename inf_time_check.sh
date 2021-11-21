#!/bin/bash



# Warm-up 3 times
for i in {1..3}
do    
    python3 demo_tf.py --checkpoint_path tf_ckpt/211102_2 --use_painted --use_tflite
    sleep 3s 
done

#Inf time check for 10 runs
for i in {1..20}
do    
    python3 demo_tf.py --checkpoint_path tf_ckpt/211102_2 --use_painted --use_tflite --inf_time_file=jetson_inf_time/2way_multithr.log
    sleep 3s 
done