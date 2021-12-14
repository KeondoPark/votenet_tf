#!/bin/bash



# Warm-up 3 times
for i in {1..3}
do    
    python3 demo_tf.py --config_path configs/inf_211209.json --gpu_mem_limit 256
    sleep 3s 
done

#Inf time check for 10 runs
for i in {1..20}
do    
    python3 demo_tf.py --config_path configs/inf_211209.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/1way_baseline_edgetpu.log
    sleep 3s 
done