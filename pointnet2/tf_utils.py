# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on Ref: https://github.com/erikwijmans/Pointnet2_PyTorch '''
import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Tuple

class SharedMLP(layers.Layer):

    def __init__(
            self,
            args: List[int],
            input_shape: List[int],
            *,
            bn: bool = False,
            activation='relu',
            #preact: bool = False,            
            name: str = "",
            data_format: str = "channels_last"
    ):
        super(SharedMLP, self).__init__()


        self.mlp_layers = []
        for i in range(len(args) - 1):
            if i == 0:
                self.mlp_layers.append(                            
                    _ConvBase(                    
                        args[i + 1],
                        bn=bn,
                        activation=activation,
                        #preact=preact,                    
                        name=name + 'layer{}'.format(i),
                        input_shape=input_shape,
                        data_format=data_format
                    )
                )
            else:
                self.mlp_layers.append(                            
                    _ConvBase(                    
                        args[i + 1],
                        bn=bn,
                        activation=activation,
                        #preact=preact,                    
                        name=name + 'layer{}'.format(i),
                        input_shape=None,
                        data_format=data_format               
                    )
                )
  

    def call(self, inputs):
        
        x = inputs
        for mlp_layer in self.mlp_layers:            
            x = mlp_layer(x)            
        
        return x

"""
class _BNBase(layers.Layer):

    def __init__(self, data_format, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        #self.add_module(name + "bn", batch_norm(in_size))
                
        self.bn_layer=batch_norm(axis=1 if data_format=="channels_first" else -1, 
                                name=name + "bn",
                                momentum=0.9, epsilon=0.001)
        # In Tensorflow, Beta and gamma is initialized as 1 and 0, respectively
        # No need to consider below
        #nn.init.constant_(self[0].weight, 1.0)
        #nn.init.constant_(self[0].bias, 0)
    
    def call(self, inputs):
        return self.bn_layer(inputs)


class BatchNorm2d(_BNBase):

    def __init__(self, data_format: str="channels_last", name: str = ""):
        super(BatchNorm2d, self).__init__(data_format, batch_norm=layers.BatchNormalization, name=name)

    def call(self, inputs):
        return super().call(inputs)
"""        

class _ConvBase(layers.Layer):

    def __init__(
            self,            
            out_size,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding='valid',
            activation='relu',
            bn=False,
            init=tf.keras.initializers.he_normal(),
            #init=tf.keras.initializers.Ones(),
            bias_init = tf.keras.initializers.Zeros(),
            bias=True,            
            #preact=False,
            name="",
            input_shape=None,
            data_format="channels_last" #(B,H,W,C)    
    ):
        super(_ConvBase, self).__init__()

        bias = bias and (not bn)
        
        if input_shape is not None:
            self.conv_unit = layers.Conv2D(
                #in_size,
                out_size,
                kernel_size=kernel_size,
                kernel_initializer=init,
                strides=stride,
                padding=padding,
                use_bias=bias if bias else None,
                bias_initializer=bias_init if bias else None,                
                data_format=data_format,
                input_shape =input_shape,
                #kernel_regularizer=tf.keras.regularizers.l2(),
                #bias_regularizer=tf.keras.regularizers.l2()
            )
        else:
            self.conv_unit = layers.Conv2D(
                #in_size,
                out_size,
                kernel_size=kernel_size,
                kernel_initializer=init,
                strides=stride,
                padding=padding,
                use_bias=bias if bias else None,
                bias_initializer=bias_init if bias else None,                
                data_format=data_format,
                #kernel_regularizer=tf.keras.regularizers.l2(),
                #bias_regularizer=tf.keras.regularizers.l2()
            )

        self.bn = bn
        if self.bn:            
            #self.bn_unit = batch_norm(data_format)
            self.bn_unit = layers.BatchNormalization(axis=1 if data_format=="channels_first" else -1, 
                                name=name + "bn")

        if activation == 'relu':
            self.act = layers.ReLU()
        elif activation =='relu6':
            self.act = layers.ReLU(6)
    
    def call(self, inputs):
        if self.bn:
            return self.act(self.bn_unit(self.conv_unit(inputs)))
        else:
            return self.act(self.conv_unit(inputs))


"""
class Conv2d(_ConvBase):

    def __init__(
            self,            
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: str = 'valid',
            activation='relu',
            bn: bool = False,
            init=tf.keras.initializers.he_normal(seed=0),
            #init=tf.constant_initializer(value=1),
            bias: bool = True,
            #preact: bool = False,
            name: str = "",
            input_shape = None,
            data_format: str = "channels_last"
    ):
        super().__init__(            
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=layers.Conv2D,
            batch_norm=BatchNorm2d,
            bias=bias,            
            #preact=preact,
            name=name,
            input_shape=input_shape,
            data_format = data_format
        )

    def call(self, inputs):
        return super().call(inputs)
"""

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1
    ):
        if not isinstance(model, tf.keras.Model):
            raise RuntimeError(
                "Class '{}' is not a keras model".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        def reset_momentum(sharedMLP):
            if sharedMLP is None: return
            print(sharedMLP.name)
            for l in sharedMLP.mlp_layers:
                if isinstance(l.bn_unit, (tf.keras.layers.BatchNormalization)):      
                    #print("Batch norm reschdule!", l.bn_unit.name)
                    l.bn_unit.momentum = self.lmbd(epoch)  

        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        for layer in self.model.layers:
            if hasattr(layer, 'sa1_mlp'):                
                print("Batch norm reschedule, 2way")
                reset_momentum(layer.sa1_mlp.mlp_module)
                reset_momentum(layer.sa2_mlp.mlp_module)
                reset_momentum(layer.sa3_mlp.mlp_module)
                reset_momentum(layer.sa4_mlp.mlp_module) 
                reset_momentum(layer.fp1.mlp)
                reset_momentum(layer.fp2.mlp)
                

            if hasattr(layer, 'sa1') and hasattr(layer.sa1, 'mlp_module'):
                print("Batch norm reschedule, 1way")
                reset_momentum(layer.sa1.mlp_module)
                reset_momentum(layer.sa2.mlp_module)
                reset_momentum(layer.sa3.mlp_module)
                reset_momentum(layer.sa4.mlp_module)
                reset_momentum(layer.fp1.mlp)
                reset_momentum(layer.fp2.mlp)                

            if hasattr(layer, 'bn1'):
                print("Batch norm reschdule!")
                layer.bn1.monemtum = self.lmbd(epoch) 
                layer.bn2.momentum = self.lmbd(epoch)

            if hasattr(layer, 'bn0'):
                layer.bn0.monemtum = self.lmbd(epoch) 
    
             


