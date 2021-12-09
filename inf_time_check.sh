#!/bin/bash



# Warm-up 3 times
for i in {1..3}
do    
    python3 demo_tf.py --config_path configs/inf_211207.json
    sleep 3s 
done

#Inf time check for 10 runs
for i in {1..20}
do    
    python3 demo_tf.py --config_path configs/inf_211207.json --inf_time_file=jetson_inf_time/1way.log
    sleep 3s 
done