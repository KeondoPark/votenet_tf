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
            #input_shape: List[int],
            *,
            bn: bool = False,
            activation='relu',
            #preact: bool = False,
            first: bool = False,
            name: str = "",
            data_format: str = "channels_last"
    ):
        super(SharedMLP, self).__init__()


        self.mlp_layers = []
        for i in range(len(args) - 1):
            if i == 0:
                self.mlp_layers.append(                            
                    Conv2d(                    
                        args[i + 1],
                        bn=bn,
                        activation=activation,
                        #preact=preact,                    
                        name=name + 'layer{}'.format(i),
                        input_shape=None,
                        data_format="channels_last"
                    )
                )
            else:
                self.mlp_layers.append(                            
                    Conv2d(                    
                        args[i + 1],
                        bn=bn,
                        activation=activation,
                        #preact=preact,                    
                        name=name + 'layer{}'.format(i),
                        input_shape=None,
                        data_format="channels_last"               
                    )
                )
  

    def call(self, inputs):

        first = True
        for mlp_layer in self.mlp_layers:
            if first:
                x = mlp_layer(inputs)
                first = False
            else:
                x = mlp_layer(x)
        
        return x

        
class _BNBase(layers.Layer):

    def __init__(self, data_format, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        #self.add_module(name + "bn", batch_norm(in_size))
                
        self.bn_layer=batch_norm(axis=1 if data_format=="channels_first" else -1, 
                                name=name + "bn")
        # In Tensorflow, Beta and gamma is initialized as 1 and 0, respectively
        # No need to consider below
        #nn.init.constant_(self[0].weight, 1.0)
        #nn.init.constant_(self[0].bias, 0)
    
    def call(self, inputs):
        return self.bn_layer(inputs)

        
# THere is only one batch normalization function in keras layer
#class BatchNorm1d(_BNBase):

#    def __init__(self, in_size: int, *, name: str = ""):
#        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, data_format: str="channels_last", name: str = ""):
        super(BatchNorm2d, self).__init__(data_format, batch_norm=layers.BatchNormalization, name=name)

    def call(self, inputs):
        return super().call(inputs)



#class BatchNorm3d(_BNBase):

#    def __init__(self, in_size: int, name: str = ""):
#        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class _ConvBase(layers.Layer):

    def __init__(
            self,            
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,            
            #preact=False,
            name="",
            input_shape=None,
            data_format="channels_last" #(B,H,W,C)    
    ):
        super(_ConvBase, self).__init__()

        bias = bias and (not bn)
        if input_shape is not None:
            self.conv_unit = conv(
                #in_size,
                out_size,
                kernel_size=kernel_size,
                kernel_initializer=init,
                strides=stride,
                padding=padding,
                use_bias=bias if bias else None,
                bias_initializer=init if bias else None,
                activation=activation if activation is not None else None,
                data_format=data_format,
                input_shape =input_shape
            )
        else:
            self.conv_unit = conv(
                #in_size,
                out_size,
                kernel_size=kernel_size,
                kernel_initializer=init,
                strides=stride,
                padding=padding,
                use_bias=bias if bias else None,
                bias_initializer=init if bias else None,
                activation=activation if activation is not None else None,
                data_format=data_format                
            )

        self.bn = bn
        if self.bn:            
            self.bn_unit = batch_norm(data_format)
    
    def call(self, inputs):
        if self.bn:
            return self.bn_unit(self.conv_unit(inputs))
        else:
            return self.conv_unit(inputs)

"""
class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation='relu',
            bn: bool = False,
            init=tf.keras.initializers.HeNormal(),
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=layers.Conv2D,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name
        )
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
            init=tf.keras.initializers.GlorotNormal(),
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
class Conv3d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int, int] = (1, 1, 1),
            stride: Tuple[int, int, int] = (1, 1, 1),
            padding: Tuple[int, int, int] = (0, 0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            batch_norm=BatchNorm3d,
            bias=bias,
            preact=preact,
            name=name
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn
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
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        for layer in self.model.layers:
            if isinstance(layer, (tf.keras.layers.BatchNormalization)):                
                layer.momentum = self.lmbd(epoch)       


