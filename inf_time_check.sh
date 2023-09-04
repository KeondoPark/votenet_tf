#!/bin/bash



# Warm-up 3 times
for i in {1..3}
do   
    #### SUN RGB-D ####
    #1way
    python3 demo_tf.py --config_path configs/inf_1way_fp.json --gpu_mem_limit 512    
    sleep 5s 
done

#### SUN RGB-D 1way####
# #Inf time check for 20 runs
# for i in {1..20}
# do    
#     #### SUN RGB-D ####
#     # 1way
#     python3 demo_tf.py --config_path configs/inf_1way.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_1way.log
#     # 2way gpu only
#     # python3 demo_tf.py --config_path configs/inf_2way.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_2way_gpuonly.log
#     # 2way tpu
#     # python3 demo_tf.py --config_path configs/inf_2way.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_2way_single.log

    
#     ########## Scannet ##########
#     #1way gpu only
#     # python3 demo_tf.py --config_path configs/inf_scannet_1way.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_scannet_1way.log
#     #2way gpu only
#     #python3 demo_tf.py --config_path configs/inf_scannet_2way.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_scannet_2way_gpuonly.log
#     #2way tpu
#     # python3 demo_tf.py --config_path configs/inf_scannet_2way.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_scannet_2way.log
    
#     sleep 10s 
# done


# #### SUN RGB-D 2way####
# #Inf time check for 20 runs
# for i in {1..20}
# do    
#     # 2way tpu
#     python3 demo_tf.py --config_path configs/inf_2way.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_2way.log    
#     sleep 10s 
# done





# #### SUN RGB-D 1way CPU + TPU ####     use_multiThr: false, use_edgetpu: true, use_tflite: true
# #Inf time check for 20 runs
# for i in {1..20}
# do    
#     python3 demo_tf.py --config_path configs/inf_1way.json --gpu_mem_limit 512 \
#                        --inf_time_file=jetson_inf_time/inf_1way_cputpu.log --run_point_cpu    
#     sleep 10s 
# done

# #### SUN RGB-D 2way CPU + TPU####     use_multiThr: true, use_edgetpu: true, use_tflite: true
#Inf time check for 20 runs
# for i in {1..20}
# do    
#     python3 demo_tf.py --config_path configs/inf_2way.json --gpu_mem_limit 256 \
#                        --inf_time_file=jetson_inf_time/inf_2way_cputpu.log --run_point_cpu
#     sleep 10s 
# done

# #### scannet 1way CPU + TPU ####    use_multiThr: false, use_edgetpu: true, use_tflite: true
# #Inf time check for 20 runs
# for i in {1..20}
# do  
#     python3 demo_tf.py --config_path configs/inf_scannet_1way.json --gpu_mem_limit 512 \
#                        --inf_time_file=jetson_inf_time/inf_scannet_1way_cputpu.log --run_point_cpu    
#     sleep 10s 
# done

#### scannet 2way CPU + TPU ####    use_multiThr: true, use_edgetpu: true, use_tflite: true
#Inf time check for 20 runs
# for i in {1..20}
# do 
#     python3 demo_tf.py --config_path configs/inf_scannet_2way.json --gpu_mem_limit 256 \
#                        --inf_time_file=jetson_inf_time/inf_scannet_2way_cputpu.log --run_point_cpu
#     sleep 10s 
# done






#### SUN RGB-D 1way CPU + CPU ####    use_multiThr: false, use_edgetpu: false, use_tflite: true
# #Inf time check for 20 runs
# for i in {1..20}
# do    
#     python3 demo_tf.py --config_path configs/inf_1way.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_1way_2cpu.log --run_point_cpu    
#     sleep 10s 
# done

# #### SUN RGB-D 2way CPU + CPU ####     use_multiThr: true, use_edgetpu: false, use_tflite: true
# #Inf time check for 20 runs
# for i in {1..20}
# do    
#     python3 demo_tf.py --config_path configs/inf_2way.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_2way_2cpu.log --run_point_cpu
#     sleep 10s 
# done

# #### scannet 1way CPU + CPU ####   use_multiThr: false, use_edgetpu: false, use_tflite: true
# #Inf time check for 20 runs
# for i in {1..20}
# do  
#     python3 demo_tf.py --config_path configs/inf_scannet_1way.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_scannet_1way_2cpu.log --run_point_cpu    
#     sleep 10s 
# done

#### scannet 2way CPU + CPU ####    use_multiThr: true, use_edgetpu: false, use_tflite: true
#Inf time check for 20 runs
# for i in {1..20}
# do 
#     python3 demo_tf.py --config_path configs/inf_scannet_2way.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_scannet_2way_2cpu.log --run_point_cpu
#     sleep 10s 
# done




### SUN RGB-D 1way CPU + GPU ####    use_multiThr: false, use_edgetpu: false, use_tflite: true
#Inf time check for 20 runs
# for i in {1..20}
# do    
#     python3 demo_tf.py --config_path configs/inf_1way.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_1way_cpugpu.log
#     sleep 10s 
# done

# #### SUN RGB-D 2way CPU + GPU ####     use_multiThr: true, use_edgetpu: false, use_tflite: true
# #Inf time check for 20 runs
# for i in {1..20}
# do    
#     python3 demo_tf.py --config_path configs/inf_2way.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_2way_cpugpu.log
#     sleep 10s 
# done

# #### scannet 1way CPU + GPU ####   use_multiThr: false, use_edgetpu: false, use_tflite: true
# #Inf time check for 20 runs
# for i in {1..20}
# do  
#     python3 demo_tf.py --config_path configs/inf_scannet_1way.json --gpu_mem_limit 512 --inf_time_file=jetson_inf_time/inf_scannet_1way_cpugpu.log
#     sleep 10s 
# done

# #### scannet 2way CPU + GPU ####    use_multiThr: true, use_edgetpu: false, use_tflite: true
# #Inf time check for 20 runs
# for i in {1..20}
# do 
#     python3 demo_tf.py --config_path configs/inf_scannet_2way.json --gpu_mem_limit 256 --inf_time_file=jetson_inf_time/inf_scannet_2way_cpugpu.log
#     sleep 10s 
# done

# #### SUN RGB-D 2way CPU + GPU ####     use_multiThr: true, use_edgetpu: false, use_tflite: true
for i in {1..20}
do  
    python3 demo_tf.py --config_path configs/inf_2way_fp16.json --gpu_mem_limit 512 \
                       --inf_time_file=jetson_inf_time/inf_2way_fp16_cpugpu.log
    sleep 10s 
done

# #### scannet 2way CPU + GPU ####     use_multiThr: true, use_edgetpu: false, use_tflite: true
for i in {1..20}
do  
    python3 demo_tf.py --config_path configs/inf_scannet_2way_fp16.json --gpu_mem_limit 512 \
                       --inf_time_file=jetson_inf_time/inf_scannet_2way_fp16_cpugpu.log
    sleep 10s 
done

# #### SUN RGB-D 2way CPU + CPU ####     use_multiThr: true, use_edgetpu: false, use_tflite: true
for i in {1..20}
do  
    python3 demo_tf.py --config_path configs/inf_2way_fp16.json --gpu_mem_limit 512 \
                       --inf_time_file=jetson_inf_time/inf_2way_fp16_2cpu.log
    sleep 10s 
done

# #### scannet 2way CPU + CPU ####     use_multiThr: true, use_edgetpu: false, use_tflite: true
for i in {1..20}
do  
    python3 demo_tf.py --config_path configs/inf_scannet_2way_fp16.json --gpu_mem_limit 512 \
                       --inf_time_file=jetson_inf_time/inf_scannet_2way_fp16_2cpu.log
    sleep 10s 
done