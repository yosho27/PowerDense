import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.models import Model, Sequential
import numpy as np
from keras.layers import Input, Dense, Lambda, concatenate
from keras.losses import Loss
import tensorflow as tf
import keras.backend as K
from random import random
from math import sqrt
from keras.regularizers import l1
from keras.optimizers import SGD

def logcopy(tensors):
    log_tensors = tf.log(tensors)
    return concatenate([tensors,log_tensors])

def logcopy_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2
    shape[-1] *= 2
    return tuple(shape)

def expcopy(tensors):
    exp_tensors = tf.exp(tensors)
    return exp_tensors

def expcopy_output_shape(input_shape):
    return input_shape

def complex_clip(t):
    v = 30
    t_real = tf.clip_by_value(tf.math.real(t),-np.exp(v),np.exp(v))
    t_imag = tf.clip_by_value(tf.math.imag(t),-np.exp(v),np.exp(v))
    return tf.cast(tf.complex(t_real,t_imag),tf.complex128)

class PowerDense(Dense):
    def __init__(self, units, use_log, copy_log = False,
                 kernel_initializer='glorot_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super().__init__(units,
                 activation=None,
                 use_bias=False,
                 kernel_initializer=kernel_initializer,
                 kernel_regularizer=kernel_regularizer,
                 kernel_constraint=kernel_constraint,
                 **kwargs)
        self.use_log = use_log
        self.copy_log = copy_log

    def compute_output_shape(self, input_shape):
        assert input_shape and (len(input_shape)==2 or len(input_shape)==3 and input_shape[2]==2)
        assert input_shape[1]
        return (input_shape[0],self.units,2)

    def call(self, inputs):
        if len(inputs.shape)==2:
            inputs = tf.cast(tf.complex(inputs,tf.zeros_like(inputs)),tf.complex128)
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.cast(tf.complex(input_real,input_imag),tf.complex128)
        if self.use_log:
            input_log = tf.math.log(inputs)
            if self.copy_log:
                inputs = tf.concat([inputs,input_log],axis=1)
            else:
                inputs = tf.concat([input_log],axis=1)
        inputs = complex_clip(inputs)
        #USE RECTIFIED LOG ON THE WEIGHTS MATRIX SO THE SMALL ONES WILL ACTUALLY BE ZERO
        outputs = tf.matmul(inputs,tf.cast(tf.complex(self.kernel,tf.zeros_like(self.kernel)),tf.complex128))
        if self.use_log:
            outputs = tf.exp(outputs)
        outputs = complex_clip(outputs)
        output_real = tf.math.real(outputs)
        output_imag = tf.math.imag(outputs)
        outputs = tf.stack((output_real,output_imag),axis=2)
        return outputs

    def build(self, input_shape):
        assert len(input_shape)==2 or len(input_shape)==3 and input_shape[2]==2
        input_dim = input_shape[1]
        if self.use_log and self.copy_log:
            input_dim*=2
        self.kernel = self.add_weight(shape=(input_dim,self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)


def l0_1(weight_matrix):
    #return tf.cast(tf.pow(tf.norm(weight_matrix,ord=0.01),0.01),tf.float64)
    return tf.cast(tf.reduce_sum(tf.pow(tf.abs(weight_matrix),0.01)),tf.float64)


log_thresh = 20

def reLogInv(weight_matrix):
    return tf.where(tf.abs(weight_matrix)>np.exp(-log_thresh),weight_matrix,tf.zeros_like(weight_matrix))

#GIVE A BONUS TO THE ONES THAT ACTUALLY ARE RECTIFIED TO ZERO
def llog(weight_matrix):
    values = tf.where(tf.abs(weight_matrix)>np.exp(-log_thresh),tf.add(tf.math.log(tf.abs(weight_matrix)),log_thresh+1),tf.zeros_like(weight_matrix))
    return 0.01*tf.cast(tf.reduce_sum(values),tf.float64)


def complex_squared_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.stack((y_true,tf.zeros_like(y_true)),axis=2)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.cast(tf.square(y_pred - y_true),tf.float64)

def complex_absolute_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.stack((y_true,tf.zeros_like(y_true)),axis=2)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.cast(tf.abs(y_pred - y_true),tf.float64)

def complex_log_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.stack((y_true,tf.zeros_like(y_true)),axis=2)
    y_true = tf.cast(y_true, y_pred.dtype)
    values = tf.nn.relu(tf.add(tf.math.log(tf.abs(y_pred - y_true)),10))
    return tf.cast(values,tf.float64)

#IF I WANT THIS TO WORK THEN THE LOSS NEEDS TO BE LOGRITHMIC
def complex_mean_absolute_error(y_true, y_pred):
    print(y_true.shape,y_pred.shape)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.stack((y_true,tf.zeros_like(y_true)),axis=2)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.abs(y_pred - y_true))

def complex_target(y):
    return tf.stack((y,tf.zeros_like(y)),axis=2)

def create_weights():
    weights = [np.zeros((4,8)),np.zeros((8,8)),np.zeros((16,8)),np.zeros((8,1))]
    sparse = [(0,2,0,1.0),(0,2,1,1.0),(0,2,2,2.0),(0,3,1,1.0),(0,3,3,2.0),(1,0,0,1.0),
              (1,1,1,1.0),(1,2,2,1.0),(1,3,2,1.0),(2,1,0,0.09531),(2,8,0,1.0),(2,10,1,0.5),(3,0,0,1.0),(3,1,0,1.0)]
    for s in sparse:
        weights[s[0]][s[1],s[2]]=s[3]
    return weights


value_in = Input(shape=(2,))
p_dense_1 = PowerDense(8, True, copy_log = True, name='power_dense_1', kernel_regularizer=l0_1)(value_in)
dense_1 = PowerDense(8, False, name='dense_1', kernel_regularizer=l0_1)(p_dense_1)
p_dense_2 = PowerDense(8, True, copy_log = True, name='power_dense_2', kernel_regularizer=l0_1)(dense_1)
output = PowerDense(1, False, name='dense_2', kernel_regularizer=l0_1)(p_dense_2)
    
model = Model(inputs=[value_in], outputs=[output])
sgd = SGD(learning_rate=0.01)
model.compile(loss=complex_absolute_error, optimizer=sgd, metrics=[])#'mae'
model.summary()

m = 1
N=10000
data = [[(random()*2-1)*m,(random()*2-1)*m] for _ in range(10000)]
target=[[x*1.1**(x*y)+np.sqrt(x*x+y*y)] for [x,y] in data]

Nt = int(N*0.8)
x_train = np.array(data[:Nt])
y_train = np.array(target[:Nt])
x_test = np.array(data[Nt:])
y_test = np.array(target[Nt:])
#y_test = complex_target(y_test)

model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
#model.evaluate(x_test,y_test,verbose=0)
model.fit(x_train,y_train,epochs=0,verbose=2,validation_data=(x_test, y_test))#, steps_per_epoch=int(Nt/32), validation_steps=1
