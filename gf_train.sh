#!/bin/bash

# sunrgbd baseline
CUDA_VISIBLE_DEVICES=0 python train_tf.py --config_path configs/config_gf_baseline.json \
                       --max_epoch 600 --lr_decay_steps 420,480,540 --lr_decay_rates 0.1,0.1,0.1 \
                       --size_cls_agnostic --size_delta 0.0625 --heading_delta 0.04 --center_delta 0.1111111111111 \
                       --num_point 20000 --num_decoder_layers 6 \
                       --learning_rate 0.004 --decoder_learning_rate 0.0002 --weight_decay 0.00000001 \
                       --query_points_generator_loss_coef 0.2 --obj_loss_coef 0.4 --optimizer adamw


# sunrgbd cosine lr schedule
CUDA_VISIBLE_DEVICES=1 python train_tf.py --config_path configs/config_gf_baseline.json \
                       --max_epoch 300 --lr-scheduler cosine \
                       --size_cls_agnostic --size_delta 0.0625 --heading_delta 0.04 --center_delta 0.1111111111111 \
                       --num_point 20000 --num_decoder_layers 6 \
                       --learning_rate 0.004 --decoder_learning_rate 0.0002 --weight_decay 0.00005 \
                       --query_points_generator_loss_coef 0.2 --obj_loss_coef 0.4 --optimizer adamw --cosine_alpha 0.001

# sunrgbd 1way
CUDA_VISIBLE_DEVICES=0 python train_tf.py --config_path configs/config_gf_1way_fp.json \
                       --max_epoch 300 --lr-scheduler cosine \
                       --size_cls_agnostic --size_delta 0.0625 --heading_delta 0.04 --center_delta 0.1111111111111 \
                       --num_point 20000 --num_decoder_layers 6 \
                       --learning_rate 0.004 --decoder_learning_rate 0.0002 --weight_decay 0.00005 \
                       --query_points_generator_loss_coef 0.2 --obj_loss_coef 0.4 --optimizer adamw --cosine_alpha 0.001

# sunrgbd 2way
CUDA_VISIBLE_DEVICES=3 python train_tf.py --config_path configs/config_gf_2way_nofp_sep.json \
                       --max_epoch 300 --lr-scheduler cosine \
                       --size_cls_agnostic --size_delta 0.0625 --heading_delta 0.04 --center_delta 0.1111111111111 \
                       --num_point 20000 --num_decoder_layers 6 \
                       --learning_rate 0.004 --decoder_learning_rate 0.0002 --weight_decay 0.00005 \
                       --query_points_generator_loss_coef 0.2 --obj_loss_coef 0.4 --optimizer adamw --cosine_alpha 0.001


# scannet baseline
CUDA_VISIBLE_DEVICES=3 python train_tf.py --config_path configs/config_gf_scannet_baseline.json \
                       --num_point 50000 --num_decoder_layers 6 --size_delta 0.1111111111 --center_delta 0.04 \
                       --learning_rate 0.006 --decoder_learning_rate 0.0006 --weight_decay 0.0005 --optimizer adamw \
                       --lr_decay_rates 0.5,0.5,0.5,0.5 --lr_decay_steps 80,160,240,320 \
                       --decoder_lr_decay_rates 0.5,0.5,0.5,0.5 --decoder_lr_decay_steps 80,160,240,320

                       --lr_decay_rates 0.7,0.7,0.7,0.7,0.7,0.7,0.7 --lr_decay_steps 50,100,150,200,250,300,350 \
                       --decoder_lr_decay_rates 0.7,0.7,0.7,0.7,0.7,0.7,0.7 --decoder_lr_decay_steps 50,100,150,200,250,300,350

#cosine lr schedule
CUDA_VISIBLE_DEVICES=1 python train_tf.py --config_path configs/config_gf_scannet_baseline.json \
                       --num_point 50000 --num_decoder_layers 6 --size_delta 0.1111111111 --center_delta 0.04 \
                       --learning_rate 0.006 --decoder_learning_rate 0.0006 --weight_decay 0.0005 --optimizer adamw \
                       --lr-scheduler cosine


                       

# scannet 1way
CUDA_VISIBLE_DEVICES=2 python train_tf.py --config_path configs/config_gf_scannet_1way_fp.json \
                       --num_point 50000 --num_decoder_layers 6 --size_delta 0.1111111111 --center_delta 0.04 \
                       --learning_rate 0.006 --decoder_learning_rate 0.0006 --weight_decay 0.0005 --optimizer adamw \
                       --lr_decay_rates 0.7,0.7,0.7,0.7,0.7,0.7,0.7 --lr_decay_steps 50,100,150,200,250,300,350 \
                       --decoder_lr_decay_rates 0.7,0.7,0.7,0.7,0.7,0.7,0.7 --decoder_lr_decay_steps 50,100,150,200,250,300,350