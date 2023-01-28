#!/bin/bash



# Warm-up 3 times
for i in {1..3}
do   
    #### SUN RGB-D ####
    #1way gpu only
    # python3 demo_tf.py --config_path configs/inf_1way_fp.json --gpu_mem_limit 512
    #2way gpu only
    # python3 demo_tf.py --config_path configs/inf_2way_nofp_sep.json --gpu_mem_limit 512
    #2way tpu
    #python3 demo_tf.py --config_path configs/inf_2way_nofp_sep.json --gpu_mem_limit 256

    ########## Scannet ##########
    #1way gpu only
    # python3 demo_tf.py --config_path configs/inf_scannet_1way_fp.json --gpu_mem_limit 512
    #2way gpu only
    #python3 demo_tf.py --config_path configs/inf_scannet_2way_nofp_sep.json --gpu_mem_limit 512
    #2way tpu
    python3 demo_tf.py --config_path configs/inf_scannet_2way_nofp_sep.json --gpu_mem_limit 256
    
    sleep 5s 
done

#Inf time check for 20 runs
for i in {1..20}
do    
    #### SUN RGB-D ####
    # 1way gpu only
    # python3 demo_tf.py --config_path configs/inf_1way_fp.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_1way_fp.log
    # 2way gpu only
    # python3 demo_tf.py --config_path configs/inf_2way_nofp_sep.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_2way_nofp_sep_gpuonly.log
    # 2way tpu
    # python3 demo_tf.py --config_path configs/inf_2way_nofp_sep.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_2way_nofp_sep_single.log

    
    ########## Scannet ##########
    #1way gpu only
    # python3 demo_tf.py --config_path configs/inf_scannet_1way_fp.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_scannet_1way_fp.log
    #2way gpu only
    #python3 demo_tf.py --config_path configs/inf_scannet_2way_nofp_sep.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_scannet_2way_nofp_sep_gpuonly.log
    #2way tpu
    python3 demo_tf.py --config_path configs/inf_scannet_2way_nofp_sep.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_scannet_2way_nofp_sep.log
    
    sleep 10s 
done