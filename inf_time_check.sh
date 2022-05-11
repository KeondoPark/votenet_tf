#!/bin/bash



# Warm-up 3 times
for i in {1..3}
do    
    python3 demo_tf.py --config_path configs/inf_scannet_2way_nofp_sep.json --gpu_mem_limit 256
    sleep 3s 
done

#Inf time check for 10 runs
for i in {1..20}
do    
    python3 demo_tf.py --config_path configs/inf_scannet_2way_nofp_sep.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_scannet_2way_nofp_sep_edgetpu.log
    sleep 10s 
done