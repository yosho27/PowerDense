Python 3.6.3 (v3.6.3:2c5fed8, Oct  3 2017, 18:11:49) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
>>> a = [2.718,10,5]
>>> log_copy(a)
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 310, in assert_input_compatibility
    K.is_keras_tensor(x)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 697, in is_keras_tensor
    str(type(x)) + '`. '
ValueError: Unexpectedly found an instance of type `<class 'list'>`. Expected a symbolic tensor instance.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    log_copy(a)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 9, in log_copy
    return concatenate([tensors,log_tensors])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\layers\merge.py", line 649, in concatenate
    return Concatenate(axis=axis, **kwargs)(inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 446, in __call__
    self.assert_input_compatibility(inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 316, in assert_input_compatibility
    str(inputs) + '. All inputs to the layer '
ValueError: Layer concatenate_1 was called with an input that isn't a symbolic tensor. Received type: <class 'list'>. Full input: [[2.718, 10, 5], array([0.99989632, 2.30258509, 1.60943791])]. All inputs to the layer should be tensors.
>>> a = np.array(a)
>>> a
array([ 2.718, 10.   ,  5.   ])
>>> tf.keras.backend.variable(a)
Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    tf.keras.backend.variable(a)
NameError: name 'tf' is not defined
>>> tensorflow
Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    tensorflow
NameError: name 'tensorflow' is not defined
>>> tf
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    tf
NameError: name 'tf' is not defined
>>> tf2
Traceback (most recent call last):
  File "<pyshell#7>", line 1, in <module>
    tf2
NameError: name 'tf2' is not defined
>>> tensorflow2
Traceback (most recent call last):
  File "<pyshell#8>", line 1, in <module>
    tensorflow2
NameError: name 'tensorflow2' is not defined
>>> import tensorflow as tf
>>> tf.keras.backend.variable(a)
<tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([ 2.718, 10.   ,  5.   ], dtype=float32)>
>>> a= tf.keras.backend.variable(a)
>>> log_copy(a)
AttributeError: 'ResourceVariable' object has no attribute 'log'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<pyshell#12>", line 1, in <module>
    log_copy(a)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 8, in log_copy
    log_tensors = np.log(tensors)
TypeError: loop of ufunc does not support argument 0 of type ResourceVariable which has no callable log method
>>> import keras.backend as K
>>> def log_copy(tensors):
    log_tensors = K(tensors)
    return concatenate([tensors,log_tensors], axis=1)

>>> log_copy(a)
Traceback (most recent call last):
  File "<pyshell#16>", line 1, in <module>
    log_copy(a)
  File "<pyshell#15>", line 2, in log_copy
    log_tensors = K(tensors)
TypeError: 'module' object is not callable
>>> a
<tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([ 2.718, 10.   ,  5.   ], dtype=float32)>
>>> log_copy
<function log_copy at 0x0000028789085F28>
>>> def log_copy(tensors):
    log_tensors = K.log(tensors)
    return concatenate([tensors,log_tensors], axis=1)

>>> log_copy(a)
Traceback (most recent call last):
  File "<pyshell#21>", line 1, in <module>
    log_copy(a)
  File "<pyshell#20>", line 3, in log_copy
    return concatenate([tensors,log_tensors], axis=1)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\layers\merge.py", line 649, in concatenate
    return Concatenate(axis=axis, **kwargs)(inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 463, in __call__
    self.build(unpack_singleton(input_shapes))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\layers\merge.py", line 356, in build
    del reduced_inputs_shapes[i][self.axis]
IndexError: list assignment index out of range
>>> def log_copy(tensors):
    log_tensors = K.log(tensors)
    return concatenate([tensors,log_tensors])

>>> log_copy(a)
<tf.Tensor 'concatenate_3/concat:0' shape=(6,) dtype=float32>
>>> a=log_copy(a)
>>> a.value_index
0
>>> a.value
Traceback (most recent call last):
  File "<pyshell#27>", line 1, in <module>
    a.value
AttributeError: 'Tensor' object has no attribute 'value'
>>> a.values
Traceback (most recent call last):
  File "<pyshell#28>", line 1, in <module>
    a.values
AttributeError: 'Tensor' object has no attribute 'values'
>>> a = tf.constant(np.array([2.718,10,5]))
>>> a
<tf.Tensor: id=35, shape=(3,), dtype=float64, numpy=array([ 2.718, 10.   ,  5.   ])>
>>> a.value
Traceback (most recent call last):
  File "<pyshell#31>", line 1, in <module>
    a.value
AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'value'
>>> log_copy(a)
<tf.Tensor 'concatenate_5/concat:0' shape=(6,) dtype=float64>
>>> log_copy(a)
<tf.Tensor 'concatenate_6/concat:0' shape=(6,) dtype=float64>
>>> log_copy(log_copy(a))
<tf.Tensor 'concatenate_8/concat:0' shape=(12,) dtype=float64>
>>> K.exp
<function exp at 0x0000028788E3F7B8>
>>> data = [[random(-10,10),random(-10,10)] for _ in range(1000)]
Traceback (most recent call last):
  File "<pyshell#36>", line 1, in <module>
    data = [[random(-10,10),random(-10,10)] for _ in range(1000)]
  File "<pyshell#36>", line 1, in <listcomp>
    data = [[random(-10,10),random(-10,10)] for _ in range(1000)]
NameError: name 'random' is not defined
>>> from random import random
>>> random()
0.1396599969715927
>>> random()
0.9646222123147934
>>> data = [[random()*20-10,random()*20-10] for _ in range(1000)]
>>> len(data)
1000
>>> len(data[0])
2
>>> 2**100
1267650600228229401496703205376
>>> 1.1**100
13780.61233982238
>>> target=[x*1.1**(x*y)+sqrt(x*x+y*y) for [x,y] in data]
Traceback (most recent call last):
  File "<pyshell#45>", line 1, in <module>
    target=[x*1.1**(x*y)+sqrt(x*x+y*y) for [x,y] in data]
  File "<pyshell#45>", line 1, in <listcomp>
    target=[x*1.1**(x*y)+sqrt(x*x+y*y) for [x,y] in data]
NameError: name 'sqrt' is not defined
>>> from math import sqrt
>>> target=[x*1.1**(x*y)+sqrt(x*x+y*y) for [x,y] in data]
>>> len(target)
1000
>>> target[0]
10.027385864611203
>>> data[0]
[8.908888773788604, -3.545313578676228]
>>> data = [[random()*20-10,random()*20-10] for _ in range(10000)]
>>> target=[x*1.1**(x*y)+sqrt(x*x+y*y) for [x,y] in data]
>>> x_train = data[:8000]
>>> y_train = target[:8000]
>>> x_test = data[8000:]
>>> y_test = target[8000:]
>>> x_train=np.array(x_train)
>>> y_train=np.array(y_train)
>>> x_test=np.array(x_test)
>>> y_test=np.array(y_test)
>>> x_train.shape
(8000, 2)
>>> y_train.shape
(8000,)
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Traceback (most recent call last):
  File "<pyshell#63>", line 1, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
NameError: name 'model' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 29, in <module>
    model = Model(inputs=[value_in], outputs=[output])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\legacy\interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\network.py", line 94, in __init__
    self._init_graph_network(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\network.py", line 241, in _init_graph_network
    self.inputs, self.outputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\network.py", line 1434, in _map_graph_network
    tensor_index=tensor_index)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\network.py", line 1421, in build_map
    node_index, tensor_index)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\network.py", line 1393, in build_map
    node = layer._inbound_nodes[node_index]
AttributeError: 'NoneType' object has no attribute '_inbound_nodes'
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 2)            0                                            
__________________________________________________________________________________________________
lambda_1 (Lambda)               multiple             0           input_1[0][0]                    
                                                                 dense_2[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 8)            40          lambda_1[0][0]                   
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 8)            0           dense_1[0][0]                    
                                                                 dense_3[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 8)            72          lambda_2[0][0]                   
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 8)            136         lambda_1[1][0]                   
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            9           lambda_2[1][0]                   
==================================================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
__________________________________________________________________________________________________
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 41, in <module>
    data = [[random()*20-10,random()*20-10] for _ in range(10000)]
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 41, in <listcomp>
    data = [[random()*20-10,random()*20-10] for _ in range(10000)]
NameError: name 'random' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 2)            0                                            
__________________________________________________________________________________________________
lambda_1 (Lambda)               multiple             0           input_1[0][0]                    
                                                                 dense_2[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 8)            40          lambda_1[0][0]                   
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 8)            0           dense_1[0][0]                    
                                                                 dense_3[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 8)            72          lambda_2[0][0]                   
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 8)            136         lambda_1[1][0]                   
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            9           lambda_2[1][0]                   
==================================================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
__________________________________________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 2/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 3/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 4/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 5/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 6/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 7/50
 - 1s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 8/50
 - 1s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 9/50
 - 1s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 10/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 11/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 51, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 216, in fit_loop
    callbacks.on_epoch_end(epoch, epoch_logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 152, in on_epoch_end
    callback.on_epoch_end(epoch, logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 611, in on_epoch_end
    self.progbar.update(self.seen, self.log_values)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\generic_utils.py", line 467, in update
    sys.stdout.write(info)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 4)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 40        
_________________________________________________________________
lambda_1 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 72        
_________________________________________________________________
lambda_4 (Lambda)            (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136       
_________________________________________________________________
lambda_3 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9         
=================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 2/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 3/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 4/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 5/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 6/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 7/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 51, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 210, in fit_loop
    verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 449, in test_loop
    batch_outs = f(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> x_train.shape
(8000, 2)
>>> x_train[:10,:]
array([[ 9.27829378,  6.43169635],
       [ 9.51494763, -6.99219874],
       [ 7.69939223,  4.58107534],
       [ 6.74322477, -1.36513468],
       [-6.83269179, -4.25083353],
       [-9.40995126, -5.61722802],
       [-2.01992372,  6.1774518 ],
       [ 9.14691948,  7.78673513],
       [ 0.7226614 ,  8.03644954],
       [-5.47269781,  3.86196644]])
>>> y_train[:10,:]
Traceback (most recent call last):
  File "<pyshell#66>", line 1, in <module>
    y_train[:10,:]
IndexError: too many indices for array
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 4)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 40        
_________________________________________________________________
lambda_1 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 72        
_________________________________________________________________
lambda_4 (Lambda)            (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136       
_________________________________________________________________
lambda_3 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9         
=================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 2/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 3/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 4/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 51, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> x_train[:10,:]
array([[-3.39362956e+00, -3.08169220e+00],
       [ 1.96039100e+00, -3.14983459e+00],
       [-9.98397048e+00,  6.89372704e+00],
       [ 9.13957655e-04, -1.89454280e+00],
       [-6.66472949e-01, -5.87654362e+00],
       [ 7.84463893e+00, -5.23989160e+00],
       [-7.60724318e+00,  3.38774700e+00],
       [ 8.94384185e+00, -7.42719456e+00],
       [-1.17276697e+00,  5.95425795e+00],
       [-2.58206765e+00,  5.50120337e+00]])
>>> y_train[:10,:]
array([[-4.61099798],
       [ 4.79836132],
       [12.11859238],
       [ 1.89545683],
       [ 4.94616253],
       [ 9.58970667],
       [ 7.67513549],
       [11.6415595 ],
       [ 5.46586146],
       [ 5.4102154 ]])
>>> x_train.shape
(8000, 2)
>>> y_train.shape
(8000, 1)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 4)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 40        
_________________________________________________________________
lambda_1 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 72        
_________________________________________________________________
lambda_4 (Lambda)            (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136       
_________________________________________________________________
lambda_3 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9         
=================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 2/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 3/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 4/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 52, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 4)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 40        
_________________________________________________________________
lambda_1 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 72        
_________________________________________________________________
lambda_4 (Lambda)            (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136       
_________________________________________________________________
lambda_3 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9         
=================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 2/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 3/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 4/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 52, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 35, in <module>
    p_dense_1 = Lambda(expcopy,expcopy_output_shape)(Dense(8, activation='linear', kernel_regularizer=l1(.001))(Lambda(logcopy,logcopy_output_shape)(value_in)))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\legacy\interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\layers\core.py", line 879, in __init__
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\regularizers.py", line 88, in get
    str(identifier))
ValueError: Could not interpret regularizer identifier: tf.Tensor(0.0009999995, shape=(), dtype=float32)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 4)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 40        
_________________________________________________________________
lambda_1 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 72        
_________________________________________________________________
lambda_4 (Lambda)            (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136       
_________________________________________________________________
lambda_3 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9         
=================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
_________________________________________________________________
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 54, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1213, in fit
    self._make_train_function()
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 333, in _make_train_function
    **self._function_kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 3009, in function
    **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3543, in function
    'eager execution. You passed: %s' % (kwargs,))
ValueError: Session keyword arguments are not support during eager execution. You passed: {'batch_size': 500}
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 4)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 40        
_________________________________________________________________
lambda_1 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 72        
_________________________________________________________________
lambda_4 (Lambda)            (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136       
_________________________________________________________________
lambda_3 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9         
=================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
_________________________________________________________________
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 2/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 3/50
 - 1s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 4/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 5/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 6/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 7/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 8/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 9/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 10/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 11/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 54, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 192, in fit_loop
    callbacks._call_batch_hook('train', 'begin', batch_index, batch_logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 88, in _call_batch_hook
    delta_t_median = np.median(self._delta_ts[hook_name])
  File "<__array_function__ internals>", line 6, in median
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3502, in median
    overwrite_input=overwrite_input)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3410, in _ureduce
    r = func(a, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3535, in _median
    part = partition(a, kth, axis=axis)
  File "<__array_function__ internals>", line 6, in partition
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\fromnumeric.py", line 745, in partition
    a.partition(kth, axis=axis, kind=kind, order=order)
KeyboardInterrupt
>>> 1+i
Traceback (most recent call last):
  File "<pyshell#71>", line 1, in <module>
    1+i
NameError: name 'i' is not defined
>>> 1+j
Traceback (most recent call last):
  File "<pyshell#72>", line 1, in <module>
    1+j
NameError: name 'j' is not defined
>>> 1.1+2.2j
(1.1+2.2j)
>>> (1+0j)
(1+0j)
>>> j
Traceback (most recent call last):
  File "<pyshell#75>", line 1, in <module>
    j
NameError: name 'j' is not defined
>>> 1+1j
(1+1j)
>>> 1+0j
(1+0j)
>>> np.log(-2)

Warning (from warnings module):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 1
    import warnings
RuntimeWarning: invalid value encountered in log
nan
>>> np.log(-2+0j)
(0.6931471805599453+3.141592653589793j)
>>> data = [[random()*20-10+0j,random()*20-10+0j] for _ in range(10000)]
>>> data[:10]
[[(3.908224834194895+0j), (1.5306025709136168+0j)], [(-1.671603930179641+0j), (-5.824859465953196+0j)], [(7.385363920803783+0j), (-8.714284543303101+0j)], [(-8.548664477889723+0j), (-2.631872547325072+0j)], [(-5.217756725383891+0j), (7.101902082887733+0j)], [(-8.975933610673918+0j), (-9.832083010143268+0j)], [(9.805228136014467+0j), (4.467949158084728+0j)], [(4.670906870221469+0j), (-8.61526636480723+0j)], [(1.6649924960899387+0j), (3.9109667427633745+0j)], [(7.860512974709664+0j), (-9.980961139372056+0j)]]
>>> target=[[x*1.1**(x*y)+sqrt(x*x+y*y)] for [x,y] in data]
Traceback (most recent call last):
  File "<pyshell#82>", line 1, in <module>
    target=[[x*1.1**(x*y)+sqrt(x*x+y*y)] for [x,y] in data]
  File "<pyshell#82>", line 1, in <listcomp>
    target=[[x*1.1**(x*y)+sqrt(x*x+y*y)] for [x,y] in data]
TypeError: can't convert complex to float
>>> target=[[x*(1.1+0j)**(x*y)+sqrt(x*x+y*y)] for [x,y] in data]
Traceback (most recent call last):
  File "<pyshell#83>", line 1, in <module>
    target=[[x*(1.1+0j)**(x*y)+sqrt(x*x+y*y)] for [x,y] in data]
  File "<pyshell#83>", line 1, in <listcomp>
    target=[[x*(1.1+0j)**(x*y)+sqrt(x*x+y*y)] for [x,y] in data]
TypeError: can't convert complex to float
>>> [x,y] = data[0]
>>> x
(3.908224834194895+0j)
>>> y
(1.5306025709136168+0j)
>>> x*x
(15.274221354617715+0j)
>>> x*x+y*y
(17.61696558470509+0j)
>>> sqrt
<built-in function sqrt>
>>> sqrt(x*x+y*y)
Traceback (most recent call last):
  File "<pyshell#90>", line 1, in <module>
    sqrt(x*x+y*y)
TypeError: can't convert complex to float
>>> real(x)
Traceback (most recent call last):
  File "<pyshell#91>", line 1, in <module>
    real(x)
NameError: name 'real' is not defined
>>> from math import real
Traceback (most recent call last):
  File "<pyshell#92>", line 1, in <module>
    from math import real
ImportError: cannot import name 'real'
>>> np.real(x)
3.908224834194895
>>> np.sqrt(x*x+y*y)
(4.197256911925345+0j)
>>> target=[[x*(1.1)**(x*y)+np.sqrt(x*x+y*y)] for [x,y] in data]
>>> target=[[x*1.1**(x*y)+np.sqrt(x*x+y*y)] for [x,y] in data]
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 4)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 40        
_________________________________________________________________
lambda_1 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 72        
_________________________________________________________________
lambda_4 (Lambda)            (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136       
_________________________________________________________________
lambda_3 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9         
=================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
_________________________________________________________________
Train on 8000 samples, validate on 2000 samples
Epoch 1/50

Warning (from warnings module):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 115
    return ops.EagerTensor(value, handle, device, dtype)
ComplexWarning: Casting complex values to real discards the imaginary part
 - 3s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Epoch 2/50
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 54, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 216, in fit_loop
    callbacks.on_epoch_end(epoch, epoch_logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 152, in on_epoch_end
    callback.on_epoch_end(epoch, logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 611, in on_epoch_end
    self.progbar.update(self.seen, self.log_values)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\generic_utils.py", line 467, in update
    sys.stdout.write(info)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 4)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 40        
_________________________________________________________________
lambda_1 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 72        
_________________________________________________________________
lambda_4 (Lambda)            (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136       
_________________________________________________________________
lambda_3 (Lambda)            (None, 8)                 0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9         
=================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
_________________________________________________________________
Train on 8000 samples, validate on 2000 samples
Epoch 1/50

Warning (from warnings module):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 115
    return ops.EagerTensor(value, handle, device, dtype)
ComplexWarning: Casting complex values to real discards the imaginary part
 - 2s - loss: nan - mse: nan - val_loss: nan - val_mse: nan
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 56, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 216, in fit_loop
    callbacks.on_epoch_end(epoch, epoch_logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 152, in on_epoch_end
    callback.on_epoch_end(epoch, logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 611, in on_epoch_end
    self.progbar.update(self.seen, self.log_values)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\generic_utils.py", line 467, in update
    sys.stdout.write(info)
KeyboardInterrupt
>>> x_train.shape
(8000, 2)
>>> type(x_train)
<class 'numpy.ndarray'>
>>> zs = np.zeros(8000,2)
Traceback (most recent call last):
  File "<pyshell#99>", line 1, in <module>
    zs = np.zeros(8000,2)
TypeError: data type not understood
>>> zs = np.zeros((8000,2))
>>> xz.shape
Traceback (most recent call last):
  File "<pyshell#101>", line 1, in <module>
    xz.shape
NameError: name 'xz' is not defined
>>> zs.shape
(8000, 2)
>>> x_train = np.concatenate((x_train,zs),axis=-1)
>>> x_train.shape
(8000, 4)
>>> x_train[:10]
array([[-9.26163598+0.j,  8.20217603+0.j,  0.        +0.j,
         0.        +0.j],
       [ 1.92780061+0.j,  1.95252627+0.j,  0.        +0.j,
         0.        +0.j],
       [ 7.37597414+0.j, -8.13931603+0.j,  0.        +0.j,
         0.        +0.j],
       [-8.48846026+0.j, -7.63158998+0.j,  0.        +0.j,
         0.        +0.j],
       [-8.99416052+0.j, -8.25355762+0.j,  0.        +0.j,
         0.        +0.j],
       [ 7.89983082+0.j,  2.30681883+0.j,  0.        +0.j,
         0.        +0.j],
       [-6.41765585+0.j,  0.29408662+0.j,  0.        +0.j,
         0.        +0.j],
       [ 3.66102248+0.j,  1.67945421+0.j,  0.        +0.j,
         0.        +0.j],
       [ 5.12931263+0.j, -7.93277709+0.j,  0.        +0.j,
         0.        +0.j],
       [ 2.60274379+0.j,  6.39889189+0.j,  0.        +0.j,
         0.        +0.j]])
>>> x_train = x_train[:2,:]
>>> x_train.shape
(2, 4)
>>> data = [[random()*20-10,random()*20-10+0j] for _ in range(10000)]
>>> x_train = np.array(data[:8000])
>>> x_train = np.concatenate((x_train,zs),axis=2)
Traceback (most recent call last):
  File "<pyshell#110>", line 1, in <module>
    x_train = np.concatenate((x_train,zs),axis=2)
  File "<__array_function__ internals>", line 6, in concatenate
numpy.AxisError: axis 2 is out of bounds for array of dimension 2
>>> x_train = np.stack((x_train,zs))
>>> x_train.shape
(2, 8000, 2)
>>> x_train = x_train[0]
>>> x_train[:10,:]
array([[-6.32385971+0.j, -8.14642583+0.j],
       [-0.6036729 +0.j, -0.17164286+0.j],
       [ 1.63515128+0.j,  6.60524495+0.j],
       [-6.70146002+0.j,  0.26757133+0.j],
       [-1.70813307+0.j,  7.64866852+0.j],
       [-4.16904275+0.j, -5.30247937+0.j],
       [ 9.15702312+0.j, -5.09768042+0.j],
       [-4.39354217+0.j, -2.88010197+0.j],
       [ 6.22422778+0.j,  9.46786493+0.j],
       [-8.74300326+0.j, -7.66724049+0.j]])
>>> x_train = np.stack((x_train,zs),axis=2)
>>> x_train.shape
(8000, 2, 2)
>>> x_train[:10,:,:]
array([[[-6.32385971+0.j,  0.        +0.j],
        [-8.14642583+0.j,  0.        +0.j]],

       [[-0.6036729 +0.j,  0.        +0.j],
        [-0.17164286+0.j,  0.        +0.j]],

       [[ 1.63515128+0.j,  0.        +0.j],
        [ 6.60524495+0.j,  0.        +0.j]],

       [[-6.70146002+0.j,  0.        +0.j],
        [ 0.26757133+0.j,  0.        +0.j]],

       [[-1.70813307+0.j,  0.        +0.j],
        [ 7.64866852+0.j,  0.        +0.j]],

       [[-4.16904275+0.j,  0.        +0.j],
        [-5.30247937+0.j,  0.        +0.j]],

       [[ 9.15702312+0.j,  0.        +0.j],
        [-5.09768042+0.j,  0.        +0.j]],

       [[-4.39354217+0.j,  0.        +0.j],
        [-2.88010197+0.j,  0.        +0.j]],

       [[ 6.22422778+0.j,  0.        +0.j],
        [ 9.46786493+0.j,  0.        +0.j]],

       [[-8.74300326+0.j,  0.        +0.j],
        [-7.66724049+0.j,  0.        +0.j]]])
>>> data = [[random()*20-10,random()*20-10] for _ in range(10000)]
>>> x_train = np.array(data[:8000])
>>> x_train = np.stack((x_train,zs),axis=2)
>>> x_train[:10,:,:]
array([[[ 2.09360902,  0.        ],
        [ 5.19157436,  0.        ]],

       [[-4.14454873,  0.        ],
        [-6.64194265,  0.        ]],

       [[-8.49555509,  0.        ],
        [ 8.08200571,  0.        ]],

       [[-6.32620972,  0.        ],
        [-3.98006879,  0.        ]],

       [[ 1.20898985,  0.        ],
        [ 5.27202068,  0.        ]],

       [[ 7.23339142,  0.        ],
        [ 7.3967925 ,  0.        ]],

       [[ 9.63153983,  0.        ],
        [-2.46116893,  0.        ]],

       [[-9.02518346,  0.        ],
        [ 1.53247161,  0.        ]],

       [[-5.26348893,  0.        ],
        [-8.29145508,  0.        ]],

       [[ 5.12080092,  0.        ],
        [ 5.99818681,  0.        ]]])
>>> x_train[:10,:,0]
array([[ 2.09360902,  5.19157436],
       [-4.14454873, -6.64194265],
       [-8.49555509,  8.08200571],
       [-6.32620972, -3.98006879],
       [ 1.20898985,  5.27202068],
       [ 7.23339142,  7.3967925 ],
       [ 9.63153983, -2.46116893],
       [-9.02518346,  1.53247161],
       [-5.26348893, -8.29145508],
       [ 5.12080092,  5.99818681]])
>>> x_train[:10,:,1]
array([[0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.]])
>>> x_train = [[[random(),random()] for _ in range(4)] for _ in range(100)]
>>> x_train.shape
Traceback (most recent call last):
  File "<pyshell#125>", line 1, in <module>
    x_train.shape
AttributeError: 'list' object has no attribute 'shape'
>>> x_train=np.array(x_train)
>>> x_train.shape
(100, 4, 2)
>>> w=[[random() for _ in range(4)]  for _ in range(4)]
>>> w=np.array(w)
>>> w.shape
(4, 4)
>>> x_out = x_train@w
Traceback (most recent call last):
  File "<pyshell#131>", line 1, in <module>
    x_out = x_train@w
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 2)
>>> x_out = wx_train
Traceback (most recent call last):
  File "<pyshell#132>", line 1, in <module>
    x_out = wx_train
NameError: name 'wx_train' is not defined
>>> x_out = w@x_train
>>> x_out.shape
(100, 4, 2)
>>> w=np.array([[random() for _ in range(5)]  for _ in range(4)])
>>> x_out = w@x_train
Traceback (most recent call last):
  File "<pyshell#136>", line 1, in <module>
    x_out = w@x_train
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 5)
>>> x_train.shape
(100, 4, 2)
>>> w.shape
(4, 5)
>>> w=np.array([[random() for _ in range(4)]  for _ in range(5)])
>>> w.shape
(5, 4)
>>> x_out = w@x_train
>>> x_out.shape
(100, 5, 2)
>>> x_train[:10,:,:]
array([[[0.63077331, 0.72815272],
        [0.09750319, 0.50311909],
        [0.31817991, 0.24037524],
        [0.97597241, 0.29272071]],

       [[0.55475293, 0.78903447],
        [0.89285268, 0.27725469],
        [0.34433313, 0.73424628],
        [0.45844323, 0.8954899 ]],

       [[0.5165975 , 0.20710949],
        [0.65947838, 0.2386298 ],
        [0.83074976, 0.47253615],
        [0.37494609, 0.12065246]],

       [[0.11262253, 0.16476856],
        [0.15301514, 0.95430731],
        [0.39663859, 0.12232742],
        [0.57037863, 0.3492801 ]],

       [[0.29110706, 0.17606033],
        [0.71412902, 0.51426861],
        [0.18880016, 0.4467034 ],
        [0.08886693, 0.08994388]],

       [[0.88309963, 0.77876521],
        [0.88190363, 0.882942  ],
        [0.65040569, 0.90746495],
        [0.29787325, 0.82792865]],

       [[0.66298355, 0.79816225],
        [0.99949807, 0.13708142],
        [0.75300842, 0.26684718],
        [0.45694022, 0.09894991]],

       [[0.83488796, 0.01017785],
        [0.82024688, 0.74832867],
        [0.30308805, 0.04842588],
        [0.74161841, 0.44512531]],

       [[0.19749395, 0.75449806],
        [0.46676788, 0.33716982],
        [0.40949797, 0.42795577],
        [0.27694221, 0.23273389]],

       [[0.16488654, 0.67934367],
        [0.37834475, 0.25412345],
        [0.71972879, 0.86367435],
        [0.87892755, 0.30681435]]])
>>> w
array([[0.68353462, 0.37433819, 0.57988185, 0.54514729],
       [0.1722806 , 0.67281585, 0.9131141 , 0.95770926],
       [0.3406925 , 0.21011794, 0.17597874, 0.5759806 ],
       [0.29383655, 0.55036199, 0.09325588, 0.79659349],
       [0.91708638, 0.37229244, 0.39159228, 0.00701252]])
>>> 0.68353462*0.63077331+0.37433819*0.09750319+0.57988185*0.31817991+0.54514729*0.97597241
1.1842100316907207
>>> x_out[0,0,0]
1.1842100398373996
>>> 0.68353462*0.72815272 +0.37433819*0.50311909 +0.57988185*0.24037524 +0.54514729*0.29272071
0.9850194229209834
>>> x_out[0,0,1]
0.9850194240629027
>>> import tensorflow as tf
>>> x_train = [[random() for _ in range(4)] for _ in range(100)]
>>> x_train = tf.keras.backend.variable(x_train)
>>> x_train
<tf.Variable 'Variable:0' shape=(100, 4) dtype=float32, numpy=
array([[0.25545835, 0.94680274, 0.52391887, 0.65324193],
       [0.9311738 , 0.52009505, 0.6273071 , 0.01590625],
       [0.6302698 , 0.3428411 , 0.08113162, 0.6112244 ],
       [0.3005857 , 0.6293267 , 0.5259164 , 0.8666958 ],
       [0.5949255 , 0.891411  , 0.15723777, 0.22467093],
       [0.99668354, 0.62549466, 0.99700594, 0.43658084],
       [0.30486992, 0.9363955 , 0.52502936, 0.99796456],
       [0.07329896, 0.13132161, 0.53236425, 0.08087205],
       [0.99213266, 0.7217554 , 0.1925885 , 0.36891243],
       [0.6679259 , 0.0290538 , 0.6636848 , 0.02890006],
       [0.27162975, 0.6650312 , 0.5040071 , 0.4915017 ],
       [0.20806015, 0.7032106 , 0.28687018, 0.5525218 ],
       [0.33139825, 0.7927755 , 0.8218323 , 0.7526015 ],
       [0.23736006, 0.5122738 , 0.87675095, 0.68728817],
       [0.28512332, 0.88900054, 0.6968747 , 0.8114294 ],
       [0.29214993, 0.76763433, 0.70852673, 0.10505179],
       [0.37460998, 0.6689293 , 0.341315  , 0.3734626 ],
       [0.81854206, 0.05605393, 0.9013067 , 0.9503632 ],
       [0.1476936 , 0.03426914, 0.4385563 , 0.59812456],
       [0.82649076, 0.5934015 , 0.57851374, 0.43484044],
       [0.9279029 , 0.79325444, 0.10824521, 0.38263336],
       [0.49040973, 0.45356506, 0.4558757 , 0.16287315],
       [0.09982553, 0.56408024, 0.47601825, 0.76135826],
       [0.61958367, 0.8703875 , 0.9643446 , 0.06874754],
       [0.85644436, 0.81075794, 0.42966804, 0.9380243 ],
       [0.24938993, 0.599854  , 0.44043726, 0.57161695],
       [0.42561156, 0.98637563, 0.90747756, 0.65268993],
       [0.03654335, 0.99881667, 0.28426754, 0.6570724 ],
       [0.583535  , 0.03681931, 0.63103074, 0.36138567],
       [0.6218473 , 0.37255523, 0.9752469 , 0.5360245 ],
       [0.47696945, 0.8534477 , 0.18157849, 0.690941  ],
       [0.9952005 , 0.6005294 , 0.9313273 , 0.77235234],
       [0.75852907, 0.5986841 , 0.6948679 , 0.07070681],
       [0.6557417 , 0.39970726, 0.8561994 , 0.5172792 ],
       [0.99862945, 0.1725565 , 0.6910536 , 0.16314422],
       [0.23771183, 0.42925453, 0.61659986, 0.00979   ],
       [0.5185278 , 0.86779547, 0.9489778 , 0.70811886],
       [0.5439594 , 0.52511704, 0.09698422, 0.9397069 ],
       [0.7750644 , 0.0321478 , 0.96388227, 0.9700495 ],
       [0.13256979, 0.61485595, 0.09848037, 0.3192711 ],
       [0.30946958, 0.73938787, 0.5577415 , 0.5134097 ],
       [0.5263392 , 0.7813401 , 0.3064519 , 0.64064157],
       [0.18962397, 0.13709246, 0.3635669 , 0.5574122 ],
       [0.04743173, 0.9374489 , 0.849054  , 0.86968166],
       [0.10738647, 0.5953454 , 0.7433441 , 0.5633113 ],
       [0.70444953, 0.37416327, 0.64313173, 0.20810704],
       [0.8958026 , 0.01509622, 0.45489353, 0.15831684],
       [0.5283448 , 0.4935133 , 0.24107148, 0.24319395],
       [0.22606005, 0.00535931, 0.14572322, 0.181281  ],
       [0.52327454, 0.5078758 , 0.8907132 , 0.50045085],
       [0.7513155 , 0.16993503, 0.95024884, 0.6213366 ],
       [0.9003562 , 0.9534045 , 0.7663498 , 0.7967614 ],
       [0.1568236 , 0.6901174 , 0.91645527, 0.65677476],
       [0.59399205, 0.06026283, 0.8676954 , 0.2753186 ],
       [0.17453927, 0.9717948 , 0.95692927, 0.5911692 ],
       [0.32877332, 0.5128138 , 0.4579757 , 0.8085928 ],
       [0.5535855 , 0.31499898, 0.74136484, 0.4084306 ],
       [0.5957696 , 0.3231277 , 0.34236228, 0.87978405],
       [0.8190116 , 0.9813276 , 0.05219363, 0.33312848],
       [0.4038879 , 0.08096891, 0.9874904 , 0.01518913],
       [0.9029545 , 0.23622438, 0.1479109 , 0.9118186 ],
       [0.5156162 , 0.05956443, 0.29886314, 0.41294724],
       [0.25080845, 0.8698428 , 0.8485996 , 0.12411949],
       [0.7394361 , 0.44638202, 0.2981686 , 0.42868918],
       [0.20439708, 0.4498621 , 0.37671947, 0.75522244],
       [0.3543033 , 0.10419279, 0.27009887, 0.84249294],
       [0.04841859, 0.71735567, 0.8151742 , 0.55644274],
       [0.9558679 , 0.93368393, 0.2398866 , 0.08978504],
       [0.9056245 , 0.67990816, 0.9159989 , 0.14814246],
       [0.79767776, 0.78891116, 0.5283721 , 0.74925977],
       [0.5119099 , 0.4478631 , 0.12335631, 0.8003546 ],
       [0.487485  , 0.5552481 , 0.94959635, 0.13688849],
       [0.32159552, 0.8586456 , 0.7949193 , 0.21254387],
       [0.92601407, 0.04626428, 0.7182894 , 0.2830111 ],
       [0.7061261 , 0.31671295, 0.18088348, 0.6831899 ],
       [0.6468782 , 0.14542966, 0.7632885 , 0.7938431 ],
       [0.01394126, 0.9484885 , 0.9735553 , 0.34699792],
       [0.66354215, 0.8271768 , 0.22555159, 0.15126981],
       [0.8243342 , 0.26487193, 0.7303167 , 0.65796745],
       [0.01057872, 0.11050916, 0.22924314, 0.12591057],
       [0.36022794, 0.23527484, 0.2940679 , 0.29290608],
       [0.92620486, 0.6235865 , 0.50282186, 0.5879015 ],
       [0.02348502, 0.9626724 , 0.8872875 , 0.86379284],
       [0.94002247, 0.6234694 , 0.45482168, 0.22496092],
       [0.33076218, 0.11104807, 0.16844171, 0.8810477 ],
       [0.92067695, 0.7479992 , 0.51458657, 0.32923204],
       [0.01589013, 0.9538984 , 0.20774044, 0.5526629 ],
       [0.26714408, 0.98095787, 0.88439745, 0.07954466],
       [0.7945742 , 0.92802143, 0.01532042, 0.2066492 ],
       [0.00296748, 0.456076  , 0.7147839 , 0.49179104],
       [0.09190849, 0.24023436, 0.751     , 0.29858834],
       [0.20240894, 0.18862963, 0.8962152 , 0.42025903],
       [0.3312815 , 0.18268627, 0.51302105, 0.7811544 ],
       [0.22015338, 0.74406636, 0.42356673, 0.2030805 ],
       [0.00560112, 0.952402  , 0.9620841 , 0.3893518 ],
       [0.3842367 , 0.64945483, 0.56964785, 0.03564888],
       [0.50770766, 0.9769982 , 0.86496586, 0.812464  ],
       [0.5189617 , 0.79880947, 0.72001255, 0.81920934],
       [0.3232627 , 0.6623837 , 0.690519  , 0.04962118],
       [0.32631385, 0.61502236, 0.6301008 , 0.5242633 ]], dtype=float32)>
>>> x_train.shape
TensorShape([100, 4])
>>> x_train_2=K.stack((x_train,K.zeros((x_train.shape))),axis=2)
>>> x_train_2.shape
TensorShape([100, 4, 2])
>>> (None,4)*(4,5)=(None,5)
SyntaxError: can't assign to operator
>>> (5,4)*(None,4,2)=(None,5,2)
SyntaxError: can't assign to operator
>>> x_real=K.slice(x_train_2,0,1)
Traceback (most recent call last):
  File "<pyshell#158>", line 1, in <module>
    x_real=K.slice(x_train_2,0,1)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 2905, in slice
    len_start = int_shape(start)[0] if is_tensor(start) else len(start)
TypeError: object of type 'int' has no len()
>>> type(x_train_2)
<class 'tensorflow.python.framework.ops.EagerTensor'>
>>> x_real=K.slice(x_train_2,[0,0,0],[:,:,1])
SyntaxError: invalid syntax
>>> x_real=K.slice(x_train_2,[0,0,0],[100,4,1])
>>> x_real.shape
TensorShape([100, 4, 1])
>>> x_real=K.squeeze(K.slice(x_train_2,[0,0,0],[100,4,1]),2)
>>> x_real.shape
TensorShape([100, 4])
>>> x_real=K.squeeze(K.slice(x_train_2,[0,0,0],[None,None,1]),2)
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_array_ops.py", line 10472, in _slice
    name, _ctx._post_execution_callbacks, input, begin, size)
tensorflow.python.eager.core._FallbackException: This function does not handle the case of the path where all inputs are not already EagerTensors.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#165>", line 1, in <module>
    x_real=K.squeeze(K.slice(x_train_2,[0,0,0],[None,None,1]),2)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 2909, in slice
    return tf.slice(x, start, size)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\array_ops.py", line 733, in slice
    return gen_array_ops._slice(input_, begin, size, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_array_ops.py", line 10477, in _slice
    input, begin, size, name=name, ctx=_ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_array_ops.py", line 10510, in _slice_eager_fallback
    _attr_Index, _inputs_Index = _execute.args_to_matching_eager([begin, size], _ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 199, in args_to_matching_eager
    accept_symbolic_tensors=False))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 254, in _constant_impl
    t = convert_to_eager_tensor(value, ctx, dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 115, in convert_to_eager_tensor
    return ops.EagerTensor(value, handle, device, dtype)
ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.
>>> x_real=K.squeeze(K.slice(x_train_2,[0,0,0],[100,4,1]),2)
>>> x_real.shape[0]
100
>>> x_real.shape[1]
4
>>> K.add
Traceback (most recent call last):
  File "<pyshell#170>", line 1, in <module>
    K.add
AttributeError: module 'keras.backend' has no attribute 'add'
>>> tf.add
<function add at 0x000001BB58F2C158>
>>> x_train_2.shape
TensorShape([100, 4, 2])
>>> x_real=tf.squeeze(tf.slice(x_train_2,[0,0,0],[100,4,1]),2)
>>> x_real.shape
TensorShape([100, 4])
>>> x_imag=tf.squeeze(tf.slice(x_train_2,[0,0,1],[100,4,1]),2)
>>> out_real=tf.log(tf.sqrt(tf.add(tf.square(x_real),tf.square(x_imag))))
Traceback (most recent call last):
  File "<pyshell#176>", line 1, in <module>
    out_real=tf.log(tf.sqrt(tf.add(tf.square(x_real),tf.square(x_imag))))
AttributeError: module 'tensorflow' has no attribute 'log'
>>> out_real=tf.math.log(tf.sqrt(tf.add(tf.square(x_real),tf.square(x_imag))))
>>> out_real.shape
TensorShape([100, 4])
>>> 1j
1j
>>> 1j*1j
(-1+0j)
>>> x_imag.shape
TensorShape([100, 4])
>>> x_complex = tf.complex(x_real,x_imag)
>>> x_complex.shape
TensorShape([100, 4])
>>> x_log = tf.math.log(x_complex)
>>> x_log.shape
TensorShape([100, 4])
>>> x_log[:10]
<tf.Tensor: id=3198, shape=(10, 4), dtype=complex64, numpy=
array([[-1.3646959e+00+0.j, -5.4664493e-02+0.j, -6.4641845e-01+0.j,
        -4.2580771e-01+0.j],
       [-7.1309328e-02+0.j, -6.5374368e-01+0.j, -4.6631902e-01+0.j,
        -4.1410427e+00+0.j],
       [-4.6160728e-01+0.j, -1.0704882e+00+0.j, -2.5116825e+00+0.j,
        -4.9229109e-01+0.j],
       [-1.2020224e+00+0.j, -4.6310478e-01+0.j, -6.4261299e-01+0.j,
        -1.4306724e-01+0.j],
       [-5.1931906e-01+0.j, -1.1494970e-01+0.j, -1.8499962e+00+0.j,
        -1.4931185e+00+0.j],
       [-3.3220053e-03+0.j, -4.6921247e-01+0.j, -2.9985905e-03+0.j,
        -8.2878172e-01+0.j],
       [-1.1878700e+00+0.j, -6.5717340e-02+0.j, -6.4430112e-01+0.j,
        -2.0375252e-03+0.j],
       [-2.6132088e+00+0.j, -2.0301061e+00+0.j, -6.3042736e-01+0.j,
        -2.5148869e+00+0.j],
       [-7.8984499e-03+0.j, -3.2606900e-01+0.j, -1.6471995e+00+0.j,
        -9.9719596e-01+0.j],
       [-4.0357804e-01+0.j, -3.5386059e+00+0.j, -4.0994799e-01+0.j,
        -3.5439117e+00+0.j]], dtype=complex64)>
>>> x_complex[:10]
<tf.Tensor: id=3203, shape=(10, 4), dtype=complex64, numpy=
array([[0.25545835+0.j, 0.94680274+0.j, 0.52391887+0.j, 0.65324193+0.j],
       [0.9311738 +0.j, 0.52009505+0.j, 0.6273071 +0.j, 0.01590625+0.j],
       [0.6302698 +0.j, 0.3428411 +0.j, 0.08113162+0.j, 0.6112244 +0.j],
       [0.3005857 +0.j, 0.6293267 +0.j, 0.5259164 +0.j, 0.8666958 +0.j],
       [0.5949255 +0.j, 0.891411  +0.j, 0.15723777+0.j, 0.22467093+0.j],
       [0.99668354+0.j, 0.62549466+0.j, 0.99700594+0.j, 0.43658084+0.j],
       [0.30486992+0.j, 0.9363955 +0.j, 0.52502936+0.j, 0.99796456+0.j],
       [0.07329896+0.j, 0.13132161+0.j, 0.53236425+0.j, 0.08087205+0.j],
       [0.99213266+0.j, 0.7217554 +0.j, 0.1925885 +0.j, 0.36891243+0.j],
       [0.6679259 +0.j, 0.0290538 +0.j, 0.6636848 +0.j, 0.02890006+0.j]],
      dtype=complex64)>
>>> x_out = tf.concat([x_complex,x_log],axis=1)
>>> x_out.shape
TensorShape([100, 8])
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 101, in <module>
    p_dense_1 = PowerDense(8, True)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 42, in __init__
    **kwargs)
TypeError: super does not take keyword arguments
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 99, in <module>
    p_dense_1 = PowerDense(8, True)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 40, in __init__
    kernel_constraint=kernel_constraint)
TypeError: super does not take keyword arguments
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 97, in <module>
    p_dense_1 = PowerDense(8, True)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 38, in __init__
    use_bias=False)
TypeError: super does not take keyword arguments
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 101, in <module>
    p_dense_1 = PowerDense(8, True)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 463, in __call__
    self.build(unpack_singleton(input_shapes))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 83, in build
    if self.use_log:
AttributeError: 'PowerDense' object has no attribute 'use_log'
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 102, in <module>
    p_dense_1 = PowerDense(8, True)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 73, in call
    outputs = tf.dot(inputs,self.kernel)
AttributeError: module 'tensorflow' has no attribute 'dot'
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 527, in _apply_op_helper
    preferred_dtype=default_dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1036, in _TensorTensorConversionFunction
    (dtype.name, t.dtype.name, str(t)))
ValueError: Tensor conversion requested dtype complex64 for Tensor with dtype float32: 'Tensor("power_dense_1/Tensordot/Reshape_1:0", shape=(4, 8), dtype=float32)'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 102, in <module>
    p_dense_1 = PowerDense(8, True)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 73, in call
    outputs = tf.tensordot(inputs,self.kernel,[[1],[0]])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 3864, in tensordot
    ab_matmul = matmul(a_reshape, b_reshape)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 2647, in matmul
    a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 6294, in mat_mul
    name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 563, in _apply_op_helper
    inferred_from[input_arg.type_attr]))
TypeError: Input 'b' of 'MatMul' Op has type float32 that does not match type complex64 of argument 'a'.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 527, in _apply_op_helper
    preferred_dtype=default_dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1036, in _TensorTensorConversionFunction
    (dtype.name, t.dtype.name, str(t)))
ValueError: Tensor conversion requested dtype complex64 for Tensor with dtype float32: 'Tensor("power_dense_1/MatMul/ReadVariableOp:0", shape=(4, 8), dtype=float32, device=/job:localhost/replica:0/task:0/device:CPU:0)'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 102, in <module>
    p_dense_1 = PowerDense(8, True)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 73, in call
    outputs = tf.matmul(inputs,self.kernel)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 2647, in matmul
    a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 6294, in mat_mul
    name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 563, in _apply_op_helper
    inferred_from[input_arg.type_attr]))
TypeError: Input 'b' of 'MatMul' Op has type float32 that does not match type complex64 of argument 'a'.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 102, in <module>
    p_dense_1 = PowerDense(8, True)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 506, in __call__
    output_shape = self.compute_output_shape(input_shape)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 48, in compute_output_shape
    return tuple(input_shape[0],self.units,2)
TypeError: tuple() takes at most 1 argument (3 given)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 527, in _apply_op_helper
    preferred_dtype=default_dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 284, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 467, in make_tensor_proto
    _AssertCompatible(values, dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 372, in _AssertCompatible
    (dtype.name, repr(mismatch), type(mismatch).__name__))
TypeError: Expected int32, got None of type '_Message' instead.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 103, in <module>
    dense_1 = PowerDense(8, False)(p_dense_1)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 67, in call
    input_real = tf.squeeze(tf.slice(inputs,[0,0,0],[inputs.shape[0],inputs.shape[1],1]),2)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\array_ops.py", line 733, in slice
    return gen_array_ops._slice(input_, begin, size, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_array_ops.py", line 10488, in _slice
    "Slice", input=input, begin=begin, size=size, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 536, in _apply_op_helper
    repr(values), type(values).__name__, err))
TypeError: Expected int32 passed to parameter 'size' of op 'Slice', got [None, 8, 1] of type 'list' instead. Error: Expected int32, got None of type '_Message' instead.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              64        
_________________________________________________________________
power_dense_3 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
power_dense_4 (PowerDense)   (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 121, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1154, in fit
    batch_size=batch_size)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 621, in _standardize_user_data
    exception_prefix='target')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 135, in standardize_input_data
    'with shape ' + str(data_shape))
ValueError: Error when checking target: expected power_dense_4 to have 3 dimensions, but got array with shape (8000, 1)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
hi
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 120, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1154, in fit
    batch_size=batch_size)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 621, in _standardize_user_data
    exception_prefix='target')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 135, in standardize_input_data
    'with shape ' + str(data_shape))
ValueError: Error when checking target: expected dense_2 to have 3 dimensions, but got array with shape (8000, 1)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 119, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1154, in fit
    batch_size=batch_size)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 621, in _standardize_user_data
    exception_prefix='target')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 135, in standardize_input_data
    'with shape ' + str(data_shape))
ValueError: Error when checking target: expected dense_2 to have 3 dimensions, but got array with shape (8000, 1)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 100, in <module>
    class ComplexMeanSquaredError(LossFunctionWrapper):
NameError: name 'LossFunctionWrapper' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 100, in <module>
    class ComplexMeanSquaredError(LossFunctionWrapper):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 102, in ComplexMeanSquaredError
    reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
NameError: name 'losses_utils' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 116, in <module>
    model.compile(loss='complex_mean_squared_error',optimizer="rmsprop", metrics=[])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 119, in compile
    self.loss, self.output_names)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 822, in prepare_loss_functions
    loss_functions = [get_loss_function(loss) for _ in output_names]
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 822, in <listcomp>
    loss_functions = [get_loss_function(loss) for _ in output_names]
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 705, in get_loss_function
    loss_fn = losses.get(loss)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 795, in get
    return deserialize(identifier)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 776, in deserialize
    printable_module_name='loss function')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\generic_utils.py", line 167, in deserialize_keras_object
    ':' + function_name)
ValueError: Unknown loss function:complex_mean_squared_error
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 116, in <module>
    model.compile(loss=ComplexMeanSquaredError,optimizer="rmsprop", metrics=[])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 229, in compile
    self.total_loss = self._prepare_total_loss(masks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 692, in _prepare_total_loss
    y_true, y_pred, sample_weight=sample_weight)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 71, in __call__
    losses = self.call(y_true, y_pred)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 132, in call
    return self.fn(y_true, y_pred, **self._fn_kwargs)
TypeError: __init__() takes from 1 to 2 positional arguments but 3 were given
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 104, in <module>
    class ComplexMeanSquaredError(Loss):
NameError: name 'Loss' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 124, in <module>
    model.compile(loss=ComplexMeanSquaredError,optimizer="rmsprop", metrics=[])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 229, in compile
    self.total_loss = self._prepare_total_loss(masks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 692, in _prepare_total_loss
    y_true, y_pred, sample_weight=sample_weight)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 73, in __call__
    losses, sample_weight, reduction=self.reduction)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\losses_utils.py", line 160, in compute_weighted_loss
    input_dtype = K.dtype(losses)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 905, in dtype
    return x.dtype.base_dtype.name
AttributeError: 'ComplexMeanSquaredError' object has no attribute 'dtype'
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 124, in <module>
    model.compile(loss=complex_mean_squared_error,optimizer="rmsprop", metrics=[])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 229, in compile
    self.total_loss = self._prepare_total_loss(masks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 692, in _prepare_total_loss
    y_true, y_pred, sample_weight=sample_weight)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 71, in __call__
    losses = self.call(y_true, y_pred)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 132, in call
    return self.fn(y_true, y_pred, **self._fn_kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 107, in complex_mean_squared_error
    y_true = tf.stack((y_true,zeros_like(y_true)),axis=2)
NameError: name 'zeros_like' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 124, in <module>
    model.compile(loss=complex_mean_squared_error,optimizer="rmsprop", metrics=[])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 229, in compile
    self.total_loss = self._prepare_total_loss(masks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 692, in _prepare_total_loss
    y_true, y_pred, sample_weight=sample_weight)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 71, in __call__
    losses = self.call(y_true, y_pred)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 132, in call
    return self.fn(y_true, y_pred, **self._fn_kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 109, in complex_mean_squared_error
    return tf.mean(tf.square(y_pred - y_true))
AttributeError: module 'tensorflow' has no attribute 'mean'
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 135, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node loss/dense_2_loss/complex_mean_squared_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_886]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
 - 2s - loss: nan - val_loss: nan
Epoch 4/50
 - 2s - loss: nan - val_loss: nan
Epoch 5/50
 - 2s - loss: nan - val_loss: nan
Epoch 6/50
 - 1s - loss: nan - val_loss: nan
Epoch 7/50
 - 1s - loss: nan - val_loss: nan
Epoch 8/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 135, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Train on 8000 samples, validate on 2000 samples
Epoch 1/5
 - 2s - loss: nan - val_loss: nan
Epoch 2/5
 - 1s - loss: nan - val_loss: nan
Epoch 3/5
 - 2s - loss: nan - val_loss: nan
Epoch 4/5
 - 2s - loss: nan - val_loss: nan
Epoch 5/5
 - 2s - loss: nan - val_loss: nan
>>> model
<keras.engine.training.Model object at 0x00000237FF8CAA20>
>>> model.predict([2,3])
Traceback (most recent call last):
  File "<pyshell#191>", line 1, in <module>
    model.predict([2,3])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1441, in predict
    x, _, _ = self._standardize_user_data(x)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 579, in _standardize_user_data
    exception_prefix='input')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 145, in standardize_input_data
    str(data_shape))
ValueError: Error when checking input: expected input_1 to have shape (2,) but got array with shape (1,)
>>> model.predict(np.arry([[2,3]]))
Traceback (most recent call last):
  File "<pyshell#192>", line 1, in <module>
    model.predict(np.arry([[2,3]]))
AttributeError: module 'numpy' has no attribute 'arry'
>>> model.predict(np.array([[2,3]]))
array([[[nan, nan]]], dtype=float32)
>>> data.shape
Traceback (most recent call last):
  File "<pyshell#194>", line 1, in <module>
    data.shape
AttributeError: 'list' object has no attribute 'shape'
>>> x_train[0]
array([-8.18206173,  4.94886665])
>>> inputs=tf.variable(array([[-8.18206173,  4.94886665]]))
Traceback (most recent call last):
  File "<pyshell#196>", line 1, in <module>
    inputs=tf.variable(array([[-8.18206173,  4.94886665]]))
AttributeError: module 'tensorflow' has no attribute 'variable'
>>> inputs=tf.Variable(array([[-8.18206173,  4.94886665]]))
Traceback (most recent call last):
  File "<pyshell#197>", line 1, in <module>
    inputs=tf.Variable(array([[-8.18206173,  4.94886665]]))
NameError: name 'array' is not defined
>>> inputs=tf.Variable(np.array([[-8.18206173,  4.94886665]]))
>>> inputs=tf.Variable(x_train[:10,:])
>>> inputs.shape
TensorShape([10, 2])
>>> if len(inputs.shape)==2:
            inputs = tf.complex(inputs,tf.zeros_like(inputs))
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.complex(input_real,input_imag)
            
SyntaxError: unindent does not match any outer indentation level
>>> if len(inputs.shape)==2:
	inputs = tf.complex(inputs,tf.zeros_like(inputs))
else:
        input_real = inputs[:,:,0]
        input_imag = inputs[:,:,1]
        inputs = tf.complex(input_real,input_imag)

        
>>> inputs.shape
TensorShape([10, 2])
>>> inputs
<tf.Tensor: id=8870, shape=(10, 2), dtype=complex128, numpy=
array([[-8.18206173+0.j,  4.94886665+0.j],
       [ 8.87321879+0.j, -0.15886583+0.j],
       [ 6.51463064+0.j,  1.17832488+0.j],
       [ 0.7388117 +0.j,  8.4549352 +0.j],
       [-4.39852749+0.j,  5.27914497+0.j],
       [ 6.43109405+0.j, -3.42196458+0.j],
       [-4.43925404+0.j,  8.51520153+0.j],
       [-2.53326045+0.j, -3.67843093+0.j],
       [ 6.32254656+0.j, -0.6586843 +0.j],
       [-2.67559898+0.j,  7.41768005+0.j]])>
>>> if True:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)

            
>>> inputs
<tf.Tensor: id=8874, shape=(10, 4), dtype=complex128, numpy=
array([[-8.18206173+0.j        ,  4.94886665+0.j        ,
         2.10194416+3.14159265j,  1.59915859+0.j        ],
       [ 8.87321879+0.j        , -0.15886583+0.j        ,
         2.18303762+0.j        , -1.83969525+3.14159265j],
       [ 6.51463064+0.j        ,  1.17832488+0.j        ,
         1.87405052+0.j        ,  0.16409384+0.j        ],
       [ 0.7388117 +0.j        ,  8.4549352 +0.j        ,
        -0.30271219+0.j        ,  2.13475032+0.j        ],
       [-4.39852749+0.j        ,  5.27914497+0.j        ,
         1.48126982+3.14159265j,  1.66376415+0.j        ],
       [ 6.43109405+0.j        , -3.42196458+0.j        ,
         1.86114467+0.j        ,  1.23021482+3.14159265j],
       [-4.43925404+0.j        ,  8.51520153+0.j        ,
         1.49048635+3.14159265j,  2.14185298+0.j        ],
       [-2.53326045+0.j        , -3.67843093+0.j        ,
         0.92950719+3.14159265j,  1.30248628+3.14159265j],
       [ 6.32254656+0.j        , -0.6586843 +0.j        ,
         1.84412206+0.j        , -0.41751092+3.14159265j],
       [-2.67559898+0.j        ,  7.41768005+0.j        ,
         0.98417327+3.14159265j,  2.00386635+0.j        ]])>
>>> weights=model.get_weights()
>>> weights.shape
Traceback (most recent call last):
  File "<pyshell#210>", line 1, in <module>
    weights.shape
AttributeError: 'list' object has no attribute 'shape'
>>> weights
[array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32), array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32), array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32), array([[nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan]], dtype=float32)]
>>> weights[0].shape
(4, 8)
>>> weights[1].shape
(8, 8)
>>> weights[2].shape
(16, 8)
>>> weights[3].shape
(8, 1)
>>> w = [[random()*2-1 for _ in range(8)] for _ in range(4)]
>>> w.shape
Traceback (most recent call last):
  File "<pyshell#217>", line 1, in <module>
    w.shape
AttributeError: 'list' object has no attribute 'shape'
>>> w = np.array(w)
>>> w.shape
(4, 8)
>>> w
array([[-0.22542889, -0.58644345,  0.20269752,  0.02566804,  0.70601657,
         0.55778799, -0.11169234, -0.7361048 ],
       [-0.77728731,  0.84678723, -0.71756226, -0.18314723, -0.50061162,
         0.18244345,  0.21353195,  0.50996186],
       [ 0.01598087,  0.83632678,  0.859874  , -0.12096975,  0.24007633,
        -0.9777958 , -0.79621736,  0.15012782],
       [ 0.81948364, -0.27782277, -0.06136105,  0.62695748,  0.24994128,
        -0.20508208,  0.173294  , -0.19847534]])
>>> inputs.shape
TensorShape([10, 4])
>>> outputs = tf.matmul(inputs,tf.complex(w,tf.zeros_like(w)))
>>> outputs
<tf.Tensor: id=8888, shape=(10, 8), dtype=complex128, numpy=
array([[-0.65814296+0.05020537j, 10.30258309+2.62739807j,
        -3.50032253+2.70137385j, -0.36805587-0.38003769j,
        -7.34980854+0.75422202j, -6.04419844-3.07183611j,
         0.57413487-2.50139062j,  8.5447549 +0.47164046j],
       [-3.34950877+2.57448377j, -3.00132454-0.87280596j,
         3.90259853-0.19277142j, -1.16063829+1.96964501j,
         6.40844943+0.78521369j,  3.16311436-0.64428435j,
        -3.08197406+0.54441916j, -5.9197656 -0.62352868j],
       [-2.22006175+0.j        , -1.30094237+0.j        ,
         2.07635638+0.j        , -0.17241273+0.j        ,
         4.50048302+0.j        ,  1.98266895+0.j        ,
        -1.93973938+0.j        , -3.94577156+0.j        ],
       [-4.99390795+0.j        ,  5.88001128+0.j        ,
        -6.30847201+0.j        , -0.15451738+0.j        ,
        -3.25013734+0.j        ,  1.81283949+0.j        ,
         2.33384331+0.j        ,  3.29871086+0.j        ],
       [-1.72475774+0.05020537j,  7.82639421+2.62739807j,
        -3.50807075+2.70137385j, -0.21584178-0.38003769j,
        -4.97677346+0.75422202j, -3.27988804-3.07183611j,
         0.72745551-2.50139062j,  5.82210344+0.47164046j],
       [ 2.24797887+2.57448377j, -5.45440541-0.87280596j,
         5.28390214-0.19277142j,  1.33794704+1.96964501j,
         7.00783244+0.78521369j,  0.89075756-0.64428435j,
        -2.71768953+0.54441916j, -6.44378832-0.62352868j],
       [-3.83898922+0.05020537j, 10.46541346+2.62739807j,
        -5.85980898+2.70137385j, -0.51093548-0.38003769j,
        -6.50382781+0.75422202j, -2.81926685-3.07183611j,
         1.49851739-2.50139062j,  7.40884271+0.47164046j],
       [ 4.51248828+2.62468914j, -1.21372292+1.75459211j,
         2.84535475+2.50860243j,  1.31283186+1.58960733j,
         0.6016392 +1.53943571j, -3.26011269-3.71612046j,
        -1.01689345-1.95697146j, -0.13008085-0.15188822j],
       [-1.2259704 +2.57448377j, -2.60729874-0.87280596j,
         3.36554306-0.19277142j, -0.20192103+1.96964501j,
         5.13194447+0.78521369j,  1.6889171 -0.64428435j,
        -2.38750428+0.54441916j, -4.63024107-0.62352868j],
       [-3.50464762+0.05020537j,  8.11665497+2.62739807j,
        -5.14167891+2.70137385j, -0.28992109-0.38003769j,
        -4.86526844+0.75422202j, -1.51238744-3.07183611j,
         1.44639774-2.50139062j,  5.50228892+0.47164046j]])>
>>> outputs.shape
TensorShape([10, 8])
>>> if True:
            outputs = tf.exp(outputs)

            
>>> outputs
<tf.Tensor: id=8890, shape=(10, 8), dtype=complex128, numpy=
array([[ 5.17159582e-01+2.59860259e-02j, -2.59548254e+04+1.46613278e+04j,
        -2.73095085e-02+1.28640853e-02j,  6.42699165e-01-2.56730308e-01j,
         4.68413853e-04+4.40081302e-04j, -2.36581335e-03-1.65299153e-04j,
        -1.42398195e+00-1.06066420e+00j,  4.57859134e+03+2.33522521e+03j],
       [-2.96067050e-02+1.88564268e-02j,  3.19547667e-02-3.80931404e-02j,
         4.86135302e+01-9.48913299e+00j, -1.21667075e-01+2.88695922e-01j,
         4.29258919e+02+4.29100574e+02j,  1.89041769e+01-1.42012814e+01j,
         3.92373138e-02+2.37563356e-02j,  2.18041859e-03-1.56826521e-03j],
       [ 1.08602403e-01+0.00000000e+00j,  2.72275089e-01+0.00000000e+00j,
         7.97535674e+00+0.00000000e+00j,  8.41631736e-01+0.00000000e+00j,
         9.00606221e+01+0.00000000e+00j,  7.26209936e+00+0.00000000e+00j,
         1.43741406e-01+0.00000000e+00j,  1.93362914e-02+0.00000000e+00j],
       [ 6.77912019e-03+0.00000000e+00j,  3.57813279e+02+0.00000000e+00j,
         1.82081330e-03+0.00000000e+00j,  8.56828603e-01+0.00000000e+00j,
         3.87688829e-02+0.00000000e+00j,  6.12782263e+00+0.00000000e+00j,
         1.03175189e+01+0.00000000e+00j,  2.70777094e+01+0.00000000e+00j],
       [ 1.77991664e-01+8.94365327e-03j, -2.18184023e+03+1.23247505e+03j,
        -2.70987259e-02+1.27647966e-02j,  7.48365014e-01-2.98939210e-01j,
         5.02603804e-03+4.72203235e-03j, -3.75409479e-02-2.62298245e-03j,
        -1.65993389e+00-1.23641487e+00j,  3.00814924e+02+1.53425048e+02j],
       [-7.98634508e+00+5.08648064e+00j,  2.74900902e-03-3.27708187e-03j,
         1.93486066e+02-3.77675721e+01j, -1.48011305e+00+3.51206440e+00j,
         7.81678310e+02+7.81389964e+02j,  1.94843425e+00-1.46371160e+00j,
         5.64814749e-02+3.41968586e-02j,  1.29109946e-03-9.28622777e-04j],
       [ 2.14882277e-02+1.07973179e-03j, -3.05446000e+04+1.72539937e+04j,
        -2.57989445e-03+1.21525374e-03j,  5.57129186e-01-2.22548830e-01j,
         1.09152703e-03+1.02550476e-03j, -5.95045911e-02-4.15758010e-03j,
        -3.58887812e+00-2.67320422e+00j,  1.47031871e+03+7.49908667e+02j],
       [-7.92401369e+01+4.50446494e+01j, -5.42968311e-02+2.92085321e-01j,
        -1.38738878e+01+1.01793353e+01j, -6.99104235e-02+3.71602640e+00j,
         5.72271264e-02+1.82421065e+00j, -3.22214713e-02+2.08593815e-02j,
        -1.36239938e-01-3.35078774e-01j,  8.67915869e-01-1.32849380e-01j],
       [-2.47531842e-01+1.57652332e-01j,  4.73869634e-02-5.64897959e-02j,
         2.84129895e+01-5.54608223e+00j, -3.17350129e-01+7.53019567e-01j,
         1.19767855e+02+1.19723675e+02j,  4.32834669e+00-3.25156020e+00j,
         7.85784872e-02+4.75755532e-02j,  7.91723006e-03-5.69446460e-03j],
       [ 3.00194902e-02+1.50840722e-03j, -2.91663170e+03+1.64754310e+03j,
        -5.29031903e-03+2.49199341e-03j,  6.94930287e-01-2.77594365e-01j,
         5.61890623e-03+5.27904022e-03j, -2.19847225e-01-1.53607047e-02j,
        -3.40661799e+00-2.53744632e+00j,  2.18476986e+02+1.11430116e+02j]])>
>>> output_real = tf.math.real(outputs)
>>> output_imag = tf.math.imag(outputs)
>>> outputs = tf.stack((output_real,output_imag),axis=2)
>>> outputs.shape
TensorShape([10, 8, 2])
>>> data.shape
Traceback (most recent call last):
  File "<pyshell#232>", line 1, in <module>
    data.shape
AttributeError: 'list' object has no attribute 'shape'
>>> x_train.shape
(8000, 2)
>>> for [x,y] in x_train:
	if x==0 or y==0:
		print([x,y])

		
>>> if 1==nan:
	print('hi')

	
Traceback (most recent call last):
  File "<pyshell#240>", line 1, in <module>
    if 1==nan:
NameError: name 'nan' is not defined
>>> float.nan
Traceback (most recent call last):
  File "<pyshell#241>", line 1, in <module>
    float.nan
AttributeError: type object 'float' has no attribute 'nan'
>>> inputs = x_train
>>> if True:
	if len(inputs.shape)==2:
            inputs = tf.complex(inputs,tf.zeros_like(inputs))
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.complex(input_real,input_imag)
            
SyntaxError: inconsistent use of tabs and spaces in indentation
>>> if len(inputs.shape)==2:
	inputs = tf.complex(inputs,tf.zeros_like(inputs))
else:
	input_real = inputs[:,:,0]
        input_imag = inputs[:,:,1]
	inputs = tf.complex(input_real,input_imag)
	
SyntaxError: inconsistent use of tabs and spaces in indentation
>>> 
>>> if len(inputs.shape)==2:
	inputs = tf.complex(inputs,tf.zeros_like(inputs))
else:
	input_real = inputs[:,:,0]
	input_imag = inputs[:,:,1]
	inputs = tf.complex(input_real,input_imag)

	
>>> if True:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)

            
>>> np.any(isnan(inputs))
Traceback (most recent call last):
  File "<pyshell#251>", line 1, in <module>
    np.any(isnan(inputs))
NameError: name 'isnan' is not defined
>>> np.any(np.isnan(inputs))
False
>>> inputs.shape
TensorShape([8000, 4])
>>> np.log(0)

Warning (from warnings module):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 1
    import warnings
RuntimeWarning: divide by zero encountered in log
-inf
>>> z=tf.Variable([0])
>>> tf.math.log(z)
Traceback (most recent call last):
  File "<pyshell#256>", line 1, in <module>
    tf.math.log(z)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 5866, in log
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InternalError: Could not find valid device for node.
Node: {{node Log}}
All kernels registered for op Log :
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_COMPLEX128]
 [Op:Log]
>>> z
<tf.Variable 'Variable:0' shape=(1,) dtype=int32, numpy=array([0])>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 3s - loss: nan - val_loss: nan
>>> model.get_weights()
[array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32), array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32), array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32), array([[nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan]], dtype=float32)]
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Train on 8000 samples, validate on 2000 samples
>>> model.get_weights()
[array([[-0.03183198,  0.37860173,  0.5560822 , -0.4877594 , -0.29233828,
         0.43072027, -0.02874434,  0.6804238 ],
       [ 0.40983492, -0.6226799 , -0.43341997, -0.44774553, -0.2809052 ,
         0.097911  ,  0.2022757 ,  0.11584431],
       [-0.5164348 ,  0.12253773, -0.27400357,  0.16681606,  0.26445025,
        -0.37057936,  0.04286855,  0.53991026],
       [ 0.36365694,  0.54044706,  0.56560916, -0.69484067,  0.4565106 ,
        -0.16508716,  0.00531322,  0.0144009 ]], dtype=float32), array([[-0.3532019 ,  0.28382486,  0.3689919 , -0.19906312,  0.5574917 ,
        -0.24132472, -0.3398885 , -0.48369348],
       [ 0.550687  , -0.05589181, -0.3362284 ,  0.05682415,  0.3075738 ,
         0.5169472 , -0.04570735, -0.42893332],
       [-0.4754532 , -0.32863343,  0.4183764 , -0.48409352,  0.28394562,
        -0.26681197, -0.44762146, -0.45944256],
       [-0.06357336, -0.6037733 , -0.01843876,  0.16391951,  0.52646476,
        -0.26619115, -0.30876854,  0.32970303],
       [ 0.510668  ,  0.38980824,  0.09569401, -0.16119573,  0.1347177 ,
         0.20988983,  0.47812933,  0.43082422],
       [-0.11291468,  0.5020452 ,  0.16145152,  0.5855909 , -0.5384319 ,
        -0.09505159,  0.26862162,  0.53699595],
       [-0.3001234 , -0.11889836, -0.30501005,  0.41774195,  0.14864504,
        -0.1679242 , -0.46524027,  0.49942774],
       [-0.2197471 ,  0.32169372, -0.5330459 ,  0.24860561, -0.02061009,
        -0.35244256, -0.29553428, -0.47304285]], dtype=float32), array([[ 0.2461189 ,  0.10965955,  0.31112218,  0.16951954, -0.43164706,
         0.30519176,  0.25790703, -0.42661154],
       [ 0.05100965, -0.38820004, -0.3910936 ,  0.25608647, -0.45730126,
         0.18923366, -0.06612337, -0.3508458 ],
       [ 0.4344107 , -0.4944687 , -0.48602593, -0.46817935,  0.3003068 ,
         0.29952788, -0.40217245, -0.28076398],
       [-0.00870693,  0.08715487, -0.38604712,  0.03677499,  0.08609247,
        -0.05090094, -0.20735872,  0.14503634],
       [-0.00739849,  0.29571044, -0.32782984,  0.27197373, -0.21561587,
         0.11176741, -0.44563746,  0.00451005],
       [-0.4898827 , -0.35635293, -0.4010738 ,  0.42062533,  0.4027989 ,
        -0.3577615 ,  0.44711757,  0.13324702],
       [-0.14294362, -0.11270607, -0.07939589, -0.4360509 , -0.3571608 ,
         0.03471529, -0.12633276, -0.03129458],
       [ 0.1178329 ,  0.35342717, -0.13789618, -0.10606575, -0.39366865,
         0.2992258 , -0.27063859, -0.2816702 ],
       [-0.28031778,  0.06511188,  0.45671344, -0.38421965,  0.08945823,
         0.31254697,  0.20583248, -0.00579703],
       [-0.02999187,  0.42737305, -0.36419106,  0.06259406, -0.43697536,
         0.40121722, -0.31767023, -0.40635765],
       [ 0.42566025, -0.01962447, -0.36881304, -0.11793125, -0.49765062,
        -0.00427604, -0.30644333, -0.09376359],
       [-0.20185769,  0.24835443,  0.20209372, -0.39244175, -0.0936712 ,
         0.05512714, -0.31367266, -0.19971311],
       [ 0.25029314, -0.25709105,  0.17878568, -0.46606433,  0.05369127,
        -0.26762342, -0.06812131,  0.14041889],
       [-0.04536331, -0.4335966 , -0.31823158,  0.15670335,  0.41148055,
         0.4593631 ,  0.34657884, -0.34397233],
       [-0.12405741, -0.15727937,  0.24521291,  0.44744742, -0.41141963,
        -0.15718651, -0.11900949,  0.46887922],
       [-0.07359409,  0.32040632,  0.10016191, -0.18224657, -0.39367914,
         0.46604562, -0.47062957,  0.3274678 ]], dtype=float32), array([[ 0.637184  ],
       [ 0.37233162],
       [-0.4767171 ],
       [-0.0254733 ],
       [ 0.5776999 ],
       [-0.26271552],
       [ 0.30326986],
       [ 0.76088774]], dtype=float32)]
>>> y_pred=model.predict(x_test[:10])
>>> x_test[:10].shape
(10, 2)
>>> y_pred.shape
(10, 1, 2)
>>> y_pred
array([[[ 2.64477181e+00, -1.33843005e+00]],

       [[ 1.41677780e+02,  4.05069800e+03]],

       [[ 1.84632236e+23,  4.97525936e+22]],

       [[            nan,             nan]],

       [[ 1.01737720e+03,  6.28220337e+02]],

       [[ 7.98567352e+01,  2.74559689e+01]],

       [[ 4.17061210e-01,  6.34136915e-01]],

       [[            nan,             nan]],

       [[-1.45491695e+23,  4.15285208e+22]],

       [[-1.23807755e+01,  3.03152649e+02]]], dtype=float32)
>>> x_test
array([[-3.16951528,  2.09339573],
       [ 4.08701411,  0.27755658],
       [-4.18005718, -8.31705429],
       ...,
       [-0.98162775, -1.07702145],
       [-5.02550923,  1.9384647 ],
       [-3.08030289,  3.21433943]])
>>> x_test[:10]
array([[-3.16951528,  2.09339573],
       [ 4.08701411,  0.27755658],
       [-4.18005718, -8.31705429],
       [-9.30818501, -5.03933213],
       [-3.99320715, -3.35874976],
       [-1.79055657,  9.36205188],
       [ 0.10020674, -0.79482892],
       [ 9.56232021,  2.29597696],
       [-7.5108015 , -5.19946583],
       [-8.27933849, -0.87061071]])
>>> inputs = x_test[3:4]
>>> inputs
array([[-9.30818501, -5.03933213]])
>>> 	if len(inputs.shape)==2:
            inputs = tf.complex(inputs,tf.zeros_like(inputs))
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.complex(input_real,input_imag)
            
SyntaxError: unexpected indent
>>> if True:
if len(inputs.shape)==2:
            inputs = tf.complex(inputs,tf.zeros_like(inputs))
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.complex(input_real,input_imag)
            
SyntaxError: expected an indented block
>>> if True:
        if len(inputs.shape)==2:
            inputs = tf.complex(inputs,tf.zeros_like(inputs))
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.complex(input_real,input_imag)

            
>>> inputs
<tf.Tensor: id=1039, shape=(1, 2), dtype=complex128, numpy=array([[-9.30818501+0.j, -5.03933213+0.j]])>
>>> if True:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)

            
>>> inputs
<tf.Tensor: id=1043, shape=(1, 4), dtype=complex128, numpy=
array([[-9.30818501+0.j        , -5.03933213+0.j        ,
         2.23089412+3.14159265j,  1.61727356+3.14159265j]])>
>>> kernel = weights[0]
Traceback (most recent call last):
  File "<pyshell#277>", line 1, in <module>
    kernel = weights[0]
NameError: name 'weights' is not defined
>>> kernel = model.get_weights[0]
Traceback (most recent call last):
  File "<pyshell#278>", line 1, in <module>
    kernel = model.get_weights[0]
TypeError: 'method' object is not subscriptable
>>> kernel = model.get_weights()[0]
>>> outputs = tf.matmul(inputs,tf.complex(kernel,tf.zeros_like(kernel)))
Traceback (most recent call last):
  File "<pyshell#280>", line 1, in <module>
    outputs = tf.matmul(inputs,tf.complex(kernel,tf.zeros_like(kernel)))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 2647, in matmul
    a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 6284, in mat_mul
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: cannot compute MatMul as input #1(zero-based) was expected to be a complex128 tensor but is a complex64 tensor [Op:MatMul] name: MatMul/
>>> inputs
<tf.Tensor: id=1043, shape=(1, 4), dtype=complex128, numpy=
array([[-9.30818501+0.j        , -5.03933213+0.j        ,
         2.23089412+3.14159265j,  1.61727356+3.14159265j]])>
>>> tf.complex(kernel,tf.zeros_like(kernel))
<tf.Tensor: id=1061, shape=(4, 8), dtype=complex64, numpy=
array([[-0.03183198+0.j,  0.37860173+0.j,  0.5560822 +0.j,
        -0.4877594 +0.j, -0.29233828+0.j,  0.43072027+0.j,
        -0.02874434+0.j,  0.6804238 +0.j],
       [ 0.40983492+0.j, -0.6226799 +0.j, -0.43341997+0.j,
        -0.44774553+0.j, -0.2809052 +0.j,  0.097911  +0.j,
         0.2022757 +0.j,  0.11584431+0.j],
       [-0.5164348 +0.j,  0.12253773+0.j, -0.27400357+0.j,
         0.16681606+0.j,  0.26445025+0.j, -0.37057936+0.j,
         0.04286855+0.j,  0.53991026+0.j],
       [ 0.36365694+0.j,  0.54044706+0.j,  0.56560916+0.j,
        -0.69484067+0.j,  0.4565106 +0.j, -0.16508716+0.j,
         0.00531322+0.j,  0.0144009 +0.j]], dtype=complex64)>
>>> kernel
array([[-0.03183198,  0.37860173,  0.5560822 , -0.4877594 , -0.29233828,
         0.43072027, -0.02874434,  0.6804238 ],
       [ 0.40983492, -0.6226799 , -0.43341997, -0.44774553, -0.2809052 ,
         0.097911  ,  0.2022757 ,  0.11584431],
       [-0.5164348 ,  0.12253773, -0.27400357,  0.16681606,  0.26445025,
        -0.37057936,  0.04286855,  0.53991026],
       [ 0.36365694,  0.54044706,  0.56560916, -0.69484067,  0.4565106 ,
        -0.16508716,  0.00531322,  0.0144009 ]], dtype=float32)
>>> inputs = tf.cast(inputs,tf.complex128)
>>> outputs = tf.matmul(inputs,tf.cast(tf.complex(self.kernel,tf.zeros_like(self.kernel)),tf.complex128))
Traceback (most recent call last):
  File "<pyshell#285>", line 1, in <module>
    outputs = tf.matmul(inputs,tf.cast(tf.complex(self.kernel,tf.zeros_like(self.kernel)),tf.complex128))
NameError: name 'self' is not defined
>>> outputs = tf.matmul(inputs,tf.cast(tf.complex(kernel,tf.zeros_like(kernel)),tf.complex128))
>>> if True:
            outputs = tf.exp(outputs)

            
>>> outputs
<tf.Tensor: id=1069, shape=(1, 8), dtype=complex128, numpy=
array([[ 8.60460107e-02-4.47927430e-02j, -1.04892129e+00+1.86631074e+00j,
         4.13958047e-02+5.39266357e-02j, -3.71016102e+01-4.20319063e+02j,
        -1.51153193e+02+1.81593932e+02j, -4.14992279e-04-3.68813200e-03j,
         5.17343244e-01+7.89125616e-02j, -5.74122233e-04+3.33213148e-03j]])>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Train on 8000 samples, validate on 2000 samples
>>> y_pred=model.predict(x_test[:10])
>>> y_pred
array([[[ 6.15120764e+03, -4.83330479e+03]],

       [[-1.56916055e+06, -5.30379723e+05]],

       [[            nan,             nan]],

       [[ 5.87677743e+13, -1.38089279e+13]],

       [[ 7.76724764e+29, -2.02094809e+29]],

       [[-9.53684894e+07, -1.59342567e+06]],

       [[ 1.96502880e+38, -3.31191626e+38]],

       [[            nan,             nan]],

       [[-2.90361286e+19, -9.84359236e+18]],

       [[-1.16508061e+00, -5.72503952e-01]]])
>>> inputs = x_test[2:3]
>>> if True:
        if len(inputs.shape)==2:
            inputs = tf.complex(inputs,tf.zeros_like(inputs))
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.complex(input_real,input_imag)

            
>>> if True:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)

            
>>> inputs = x_test[2:3]
>>> def call(kernel, use_log, inputs):
        if len(inputs.shape)==2:
            inputs = tf.cast(tf.complex(inputs,tf.zeros_like(inputs)),tf.complex128)
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.cast(tf.complex(input_real,input_imag),tf.complex128)
        if use_log:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)
        outputs = tf.matmul(inputs,tf.cast(tf.complex(kernel,tf.zeros_like(kernel)),tf.complex128))
        if use_log:
            outputs = tf.exp(outputs)
        output_real = tf.math.real(outputs)
        output_imag = tf.math.imag(outputs)
        outputs = tf.stack((output_real,output_imag),axis=2)
        return outputs

>>> x_test
array([[ 2.69879645,  7.16476071],
       [ 6.75566744,  6.19770929],
       [-5.16861107,  8.141235  ],
       ...,
       [ 5.86960918, -6.39552484],
       [-1.83320962,  7.33531612],
       [-8.2766394 , -4.97170838]])
>>> outputs = call(model.get_weights()[0],True,inputs)
>>> outputs
<tf.Tensor: id=1111, shape=(1, 8, 2), dtype=float64, numpy=
array([[[ 1.24134571e+00,  2.63921029e-01],
        [ 1.59430342e-03,  2.66425960e-04],
        [ 4.65252203e-01,  2.96844045e-01],
        [ 1.08940927e+01, -1.00038411e+01],
        [ 1.10876190e+00,  3.75801973e-01],
        [ 3.43619891e-04, -3.53805901e-03],
        [ 2.41714681e+03,  6.38551151e+03],
        [-1.39076240e+00,  9.53464298e+00]]])>
>>> outputs = call(model.get_weights()[1],False,outputs)
>>> outputs
<tf.Tensor: id=1138, shape=(1, 8, 2), dtype=float64, numpy=
array([[[  443.99637382,  1153.18996795],
        [ 1253.11813338,  3316.7409361 ],
        [-1090.81322553, -2856.65357578],
        [   64.20660484,   161.01047342],
        [  567.84779587,  1507.32936061],
        [ -277.37508626,  -746.59681661],
        [ -619.73668558, -1640.40706302],
        [ 1343.70192833,  3533.57437977]]])>
>>> outputs = call(model.get_weights()[2],True,outputs)
>>> outputs
<tf.Tensor: id=1169, shape=(1, 8, 2), dtype=float64, numpy=
array([[[ 1.60548640e-049,  2.13414474e-049],
        [ 9.11908135e-244, -9.62785899e-246],
        [            -inf,              inf],
        [ 1.63674818e-066, -3.73622511e-066],
        [ 5.71449656e-147, -1.15600543e-147],
        [-0.00000000e+000, -0.00000000e+000],
        [ 5.14033808e+123,  2.23187734e+123],
        [            -inf,             -inf]]])>
>>> np.exp(1000)

Warning (from warnings module):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 1
    import warnings
RuntimeWarning: overflow encountered in exp
inf
>>> np.exp(5000)
inf
>>> np.exp(500)
1.4035922178528375e+217
>>> np.exp(750)
inf
>>> np.exp(600)
3.7730203009299397e+260
>>> np.exp(650)
1.956199921370272e+282
>>> np.exp(700)
1.0142320547350045e+304
>>> np.exp(720)
inf
>>> np.exp(710)
inf
>>> np.exp(700)
1.0142320547350045e+304
>>> np.exp(703)
2.037139538406043e+305
>>> np.exp(706)
4.0917041416340054e+306
>>> np.exp(708)
3.023383144276055e+307
>>> np.exp(709)
8.218407461554972e+307
>>> np.exp(710)
inf
>>> tf.constant([[1, exp(800), 13], [-exp(800), 21, 13]])
Traceback (most recent call last):
  File "<pyshell#322>", line 1, in <module>
    tf.constant([[1, exp(800), 13], [-exp(800), 21, 13]])
NameError: name 'exp' is not defined
>>> t=tf.constant([[1, np.exp(800), 13], [-np.exp(800), 21, 13]])
>>> t
<tf.Tensor: id=1171, shape=(2, 3), dtype=float32, numpy=
array([[  1.,  inf,  13.],
       [-inf,  21.,  13.]], dtype=float32)>
>>> tf.clip(t,-exp(700),exp(700))
Traceback (most recent call last):
  File "<pyshell#325>", line 1, in <module>
    tf.clip(t,-exp(700),exp(700))
AttributeError: module 'tensorflow' has no attribute 'clip'
>>> tf.clip_by_value(t,-exp(700),exp(700))
Traceback (most recent call last):
  File "<pyshell#326>", line 1, in <module>
    tf.clip_by_value(t,-exp(700),exp(700))
NameError: name 'exp' is not defined
>>> tf.clip_by_value(t,-np.exp(700),np.exp(700))
<tf.Tensor: id=1176, shape=(2, 3), dtype=float32, numpy=
array([[  1.,  inf,  13.],
       [-inf,  21.,  13.]], dtype=float32)>
>>> tf.clip_by_value(t,-np.exp(300),np.exp(300))
<tf.Tensor: id=1181, shape=(2, 3), dtype=float32, numpy=
array([[  1.,  inf,  13.],
       [-inf,  21.,  13.]], dtype=float32)>
>>> tf.clip_by_value(t,-100,100)
<tf.Tensor: id=1186, shape=(2, 3), dtype=float32, numpy=
array([[   1.,  100.,   13.],
       [-100.,   21.,   13.]], dtype=float32)>
>>> tf.clip_by_value(t,-100,np.exp(100))
<tf.Tensor: id=1191, shape=(2, 3), dtype=float32, numpy=
array([[   1.,   inf,   13.],
       [-100.,   21.,   13.]], dtype=float32)>
>>> np.exp(100)
2.6881171418161356e+43
>>> np.exp(200)
7.225973768125749e+86
>>> np.exp(300)
1.9424263952412558e+130
>>> tf.clip_by_value(t,-100,np.exp(200))
<tf.Tensor: id=1196, shape=(2, 3), dtype=float32, numpy=
array([[   1.,   inf,   13.],
       [-100.,   21.,   13.]], dtype=float32)>
>>> tf.clip_by_value(t,-100,np.exp(50))
<tf.Tensor: id=1201, shape=(2, 3), dtype=float32, numpy=
array([[ 1.0000000e+00,  5.1847055e+21,  1.3000000e+01],
       [-1.0000000e+02,  2.1000000e+01,  1.3000000e+01]], dtype=float32)>
>>> tf.clip_by_value(t,-100,np.exp(60))
<tf.Tensor: id=1206, shape=(2, 3), dtype=float32, numpy=
array([[ 1.0000000e+00,  1.1420074e+26,  1.3000000e+01],
       [-1.0000000e+02,  2.1000000e+01,  1.3000000e+01]], dtype=float32)>
>>> tf.clip_by_value(t,-100,np.exp(70))
<tf.Tensor: id=1211, shape=(2, 3), dtype=float32, numpy=
array([[ 1.0000000e+00,  2.5154387e+30,  1.3000000e+01],
       [-1.0000000e+02,  2.1000000e+01,  1.3000000e+01]], dtype=float32)>
>>> tf.clip_by_value(t,-100,np.exp(80))
<tf.Tensor: id=1216, shape=(2, 3), dtype=float32, numpy=
array([[ 1.0000000e+00,  5.5406225e+34,  1.3000000e+01],
       [-1.0000000e+02,  2.1000000e+01,  1.3000000e+01]], dtype=float32)>
>>> tf.clip_by_value(t,-100,np.exp(90))
<tf.Tensor: id=1221, shape=(2, 3), dtype=float32, numpy=
array([[   1.,   inf,   13.],
       [-100.,   21.,   13.]], dtype=float32)>
>>> tf.cast(exp(100),tf.float32)
Traceback (most recent call last):
  File "<pyshell#340>", line 1, in <module>
    tf.cast(exp(100),tf.float32)
NameError: name 'exp' is not defined
>>> tf.cast(np.exp(100),tf.float32)
<tf.Tensor: id=1224, shape=(), dtype=float32, numpy=inf>
>>> tf.cast(np.exp(70),tf.float32)
<tf.Tensor: id=1227, shape=(), dtype=float32, numpy=2.5154387e+30>
>>> tf.cast(np.exp(100),tf.complex128)
<tf.Tensor: id=1230, shape=(), dtype=complex128, numpy=(2.6881171418161356e+43+0j)>
>>> tf.cast(np.exp(200),tf.complex128)
<tf.Tensor: id=1233, shape=(), dtype=complex128, numpy=(7.225973768125749e+86+0j)>
>>> tf.cast(np.exp(300),tf.complex128)
<tf.Tensor: id=1236, shape=(), dtype=complex128, numpy=(1.9424263952412558e+130+0j)>
>>> tf.cast(np.exp(400),tf.complex128)
<tf.Tensor: id=1239, shape=(), dtype=complex128, numpy=(5.221469689764144e+173+0j)>
>>> tf.cast(np.exp(700),tf.complex128)
<tf.Tensor: id=1242, shape=(), dtype=complex128, numpy=(1.0142320547350045e+304+0j)>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 107, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1')(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 60, in call
    tf.clip_by_value(t,-np.exp(700),np.exp(700))
NameError: name 't' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 107, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1')(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 60, in call
    inputs = tf.clip_by_value(inputs,-np.exp(700),np.exp(700))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\clip_ops.py", line 72, in clip_by_value
    t_min = math_ops.minimum(values, clip_value_max)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 6728, in minimum
    "Minimum", x=x, y=y, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 626, in _apply_op_helper
    param_name=input_name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 60, in _SatisfiesTypeConstraint
    ", ".join(dtypes.as_dtype(x).name for x in allowed_list)))
TypeError: Value passed to parameter 'x' has DataType complex128 not in list of allowed values: bfloat16, float16, float32, float64, int32, int64
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 111, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1')(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 64, in call
    inputs = tf.complex_clip(inputs)
AttributeError: module 'tensorflow' has no attribute 'complex_clip'
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> y_pred=model.predict(x_test[:10])
>>> y_pred
array([[[ 2.38681765e+040, -7.16978636e+039]],

       [[-4.32533749e+035,  4.87563244e+034]],

       [[-4.73905220e+001,  3.66577501e+001]],

       [[ 2.60928936e-001,  5.04307614e+000]],

       [[ 4.66762150e+000,  2.54810416e+000]],

       [[-3.78198558e+000,  6.82793007e+000]],

       [[-1.49385324e+303, -1.49385324e+303]],

       [[ 3.65656263e+160,  2.08325817e+160]],

       [[-1.56824804e+006, -1.45933003e+008]],

       [[-5.31606471e+002, -6.81886463e+002]]])
>>> model.predict(x_test[10:30])
array([[[-1.49385324e+303, -1.49385324e+303]],

       [[-2.59445306e+000, -3.84581973e+000]],

       [[-5.73803520e+000, -4.42870181e+000]],

       [[-5.48477678e+003, -2.80682097e+003]],

       [[-1.49385324e+303,  1.46294238e+304]],

       [[ 1.00530083e+016, -3.36545441e+015]],

       [[-3.19165558e+098, -5.66981812e+099]],

       [[ 8.87704083e+013, -4.56255752e+013]],

       [[ 3.39202194e+153,  3.81726554e+154]],

       [[ 5.14910862e+031,  4.14916272e+031]],

       [[ 3.07469921e+001,  7.83308305e+001]],

       [[-1.30803923e+009,  1.04993470e+008]],

       [[ 1.47189677e+011, -1.28283730e+012]],

       [[-3.96950542e+000,  2.20621674e+000]],

       [[ 1.13686248e+032, -3.80588432e+031]],

       [[-1.08548675e+066, -6.71861019e+066]],

       [[ 3.88001640e+003,  4.81361014e+002]],

       [[ 2.59217007e+000, -1.60096302e+001]],

       [[-6.87395231e+000,  2.62320904e+000]],

       [[-3.42326057e+001,  9.02944871e+001]]])
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
 - 2s - loss: nan - val_loss: nan
Epoch 4/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 128, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> def call(kernel, use_log inputs):
        if len(inputs.shape)==2:
            inputs = tf.cast(tf.complex(inputs,tf.zeros_like(inputs)),tf.complex128)
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.cast(tf.complex(input_real,input_imag),tf.complex128)
        if use_log:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)
            inputs = complex_clip(inputs)
        outputs = tf.matmul(inputs,tf.cast(tf.complex(kernel,tf.zeros_like(self.kernel)),tf.complex128))
        if use_log:
            outputs = tf.exp(outputs)
            outputs = complex_clip(outputs)
        output_real = tf.math.real(outputs)
        output_imag = tf.math.imag(outputs)
        outputs = tf.stack((output_real,output_imag),axis=2)
        return outputs
SyntaxError: invalid syntax
>>> 
>>> def call(kernel, use_log, inputs):
        if len(inputs.shape)==2:
            inputs = tf.cast(tf.complex(inputs,tf.zeros_like(inputs)),tf.complex128)
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.cast(tf.complex(input_real,input_imag),tf.complex128)
        if use_log:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)
            inputs = complex_clip(inputs)
        outputs = tf.matmul(inputs,tf.cast(tf.complex(kernel,tf.zeros_like(self.kernel)),tf.complex128))
        if use_log:
            outputs = tf.exp(outputs)
            outputs = complex_clip(outputs)
        output_real = tf.math.real(outputs)
        output_imag = tf.math.imag(outputs)
        outputs = tf.stack((output_real,output_imag),axis=2)
        return outputs

>>> #outputs = call(model.get_weights()[2],True,outputs)
>>> y_pred=model.predict(x_train)
>>> y_isnan = np.isnan(y_pred)
>>> y_isnan.shape
(8000, 1, 2)
>>> y_isnan[:10]
array([[[False, False]],

       [[False, False]],

       [[False, False]],

       [[False, False]],

       [[False, False]],

       [[False, False]],

       [[False, False]],

       [[False, False]],

       [[False, False]],

       [[False, False]]])
>>> any(y_isnan)
Traceback (most recent call last):
  File "<pyshell#360>", line 1, in <module>
    any(y_isnan)
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
>>> y_isnan.any()
False
>>> y_isnan.all()
False
>>> x_train.shape
(8000, 2)
>>> y_pred[:10]
array([[[ 6.44847696e+019, -5.61829261e+020]],

       [[-5.89734400e+020, -1.79439728e+021]],

       [[ 7.38905079e+021, -5.53089847e+022]],

       [[-6.47021095e-001,  2.13412733e-002]],

       [[ 7.64374558e+000,  5.17134658e-001]],

       [[ 1.19440356e+023,  1.27833613e+022]],

       [[-1.14039711e+001,  1.14034770e+001]],

       [[ 3.93055394e+002,  8.49991453e+001]],

       [[-7.71676291e+005,  1.79458539e+007]],

       [[ 1.15255113e+142, -2.93099224e+142]]])
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> y_pred=model.predict(x_train)
>>> y_mse=complex_mean_squared_error(y_train, y_pred)
>>> y_mes.shape
Traceback (most recent call last):
  File "<pyshell#367>", line 1, in <module>
    y_mes.shape
NameError: name 'y_mes' is not defined
>>> y_mse.shape
TensorShape([8000, 1, 2])
>>> np.isnan(y_mse).any()
False
>>> np.isnan(y_mse).all()
False
>>> y_mse[:100]
<tf.Tensor: id=3100, shape=(100, 1, 2), dtype=float64, numpy=
array([[[2.93857335e+014, 2.26772704e+011]],

       [[9.24311659e+009, 2.40130442e+010]],

       [[2.35788094e+000, 3.94223874e+001]],

       [[3.97227655e+051, 2.31505754e+051]],

       [[1.18573073e+029, 6.52019071e+028]],

       [[1.36646542e+102, 6.15903903e+101]],

       [[3.22892562e+009, 1.27571866e+010]],

       [[5.28465344e+235, 1.88026128e+237]],

       [[6.23255375e+115, 4.07234365e+115]],

       [[8.85159269e+033, 4.86738441e+033]],

       [[2.45860150e+005, 3.19758905e+005]],

       [[1.59879937e+002, 1.41842290e+002]],

       [[2.75382195e+004, 2.36671255e+005]],

       [[            inf,             inf]],

       [[3.30955433e+003, 9.31026408e+003]],

       [[            inf,             inf]],

       [[            inf,             inf]],

       [[1.08210132e+004, 3.66177875e+003]],

       [[7.25121194e+011, 4.22595047e+011]],

       [[5.43202466e+058, 3.16580417e+058]],

       [[2.19891853e+015, 1.04408889e+016]],

       [[2.80827457e+146, 5.35126015e+146]],

       [[            inf,             inf]],

       [[1.37665281e+054, 1.75560827e+053]],

       [[1.24918029e+009, 7.27663065e+008]],

       [[7.28234901e+015, 5.18867530e+016]],

       [[            inf,             inf]],

       [[9.78672120e+286, 2.00522939e+287]],

       [[1.16566229e+013, 6.41127953e+012]],

       [[4.87330605e+011, 3.76128080e+010]],

       [[3.00759792e+281, 2.09425382e+280]],

       [[2.95078865e+017, 1.94062293e+019]],

       [[1.96169974e+104, 1.74875174e+102]],

       [[3.66955885e+163, 1.21041396e+162]],

       [[9.75249844e+051, 5.36278165e+051]],

       [[1.09345335e+170, 5.79657395e+170]],

       [[1.71680326e+002, 1.91430217e+001]],

       [[            inf,             inf]],

       [[6.24664449e+071, 8.21347042e+072]],

       [[8.99554391e+006, 7.69992488e+006]],

       [[7.64190140e+008, 5.47706226e+008]],

       [[2.77685277e+028, 5.99953732e+031]],

       [[3.66754390e+012, 1.41105882e+013]],

       [[            inf,             inf]],

       [[9.09131172e+035, 4.99920302e+035]],

       [[4.71252165e+044, 1.22914039e+043]],

       [[3.18371253e+018, 1.85547943e+018]],

       [[1.05431638e+028, 7.44671133e+027]],

       [[9.90139118e+004, 1.51069211e+004]],

       [[            inf,             inf]],

       [[2.84534265e+280, 4.04876522e+278]],

       [[3.18041022e+118, 8.55641856e+117]],

       [[7.67608632e+002, 4.19980837e+002]],

       [[1.52468634e+010, 1.21654936e+010]],

       [[3.10034930e+277, 1.80689510e+277]],

       [[            inf,             inf]],

       [[6.71685892e+003, 1.21374458e+003]],

       [[2.80820230e+002, 4.49020173e+001]],

       [[2.64327007e+006, 4.36222266e+007]],

       [[2.88589614e+023, 1.64467327e+022]],

       [[2.21810911e+010, 3.01167999e+011]],

       [[3.05392514e+043, 1.77983893e+043]],

       [[1.15958304e+067, 6.39479710e+066]],

       [[2.52592484e+054, 1.20194289e+053]],

       [[7.01762789e+121, 3.63219350e+122]],

       [[1.32640075e+021, 8.13128915e+021]],

       [[1.76825990e+004, 7.66081367e+000]],

       [[3.27401594e+002, 5.06942978e+000]],

       [[2.68987114e+165, 5.97859183e+163]],

       [[2.43961844e+014, 1.34161071e+014]],

       [[4.28543065e+004, 7.61077903e+004]],

       [[8.54362182e+000, 5.33158798e+000]],

       [[9.37286025e+002, 7.16589595e+001]],

       [[2.90856370e+055, 2.28653131e+055]],

       [[2.77534540e+004, 1.09158396e+001]],

       [[5.32530830e+045, 2.92832301e+045]],

       [[4.79251020e+012, 2.13362064e+012]],

       [[4.91260792e+014, 1.96569781e+015]],

       [[1.69078117e+004, 1.31914469e+004]],

       [[1.48987219e+017, 8.19264330e+016]],

       [[1.85923847e+079, 7.67381741e+077]],

       [[6.57065365e+140, 2.73555294e+141]],

       [[2.49758407e+002, 2.31537443e+002]],

       [[3.13747681e+245, 2.03165739e+246]],

       [[2.49845840e+104, 2.38442072e+105]],

       [[3.27790624e+002, 2.67319687e-002]],

       [[1.49947880e+014, 1.97915108e+015]],

       [[9.97154549e+056, 5.48323298e+056]],

       [[2.71273263e+005, 4.47155195e+005]],

       [[7.91636706e+002, 1.19897594e+001]],

       [[8.24806515e+009, 9.23346474e+009]],

       [[9.06283016e+034, 2.21527891e+036]],

       [[1.13102726e+015, 8.72589624e+014]],

       [[            inf,             inf]],

       [[4.21925475e+061, 2.45899736e+061]],

       [[9.49265756e+003, 1.70735541e+004]],

       [[            inf,             inf]],

       [[3.72911127e+011, 5.39815249e+010]],

       [[            inf,             inf]],

       [[4.36017649e+055, 8.04383075e+050]]])>
>>> (y_pred==float('inf')).any()
False
>>> (y_mse==float('inf')).any()
Traceback (most recent call last):
  File "<pyshell#373>", line 1, in <module>
    (y_mse==float('inf')).any()
AttributeError: 'bool' object has no attribute 'any'
>>> (y_mse==float('inf'))
False
>>> y_mse.shape
TensorShape([8000, 1, 2])
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 6, in <module>
    from . import conv_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\conv_utils.py", line 9, in <module>
    from .. import backend as K
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\__init__.py", line 1, in <module>
    from .load_backend import epsilon
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\load_backend.py", line 90, in <module>
    from .tensorflow_backend import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 5, in <module>
    import tensorflow as tf
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\__init__.py", line 40, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\__init__.py", line 73, in <module>
    from tensorflow.python.ops.standard_ops import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\standard_ops.py", line 75, in <module>
    from tensorflow.python.ops.partitioned_variables import *
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
  File "<frozen importlib._bootstrap_external>", line 758, in get_code
  File "<frozen importlib._bootstrap_external>", line 842, in path_stats
  File "<frozen importlib._bootstrap_external>", line 82, in _path_stat
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: nan - val_loss: nan
Epoch 2/50
 - 3s - loss: nan - val_loss: nan
Epoch 3/50
 - 3s - loss: nan - val_loss: nan
Epoch 4/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 128, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> y_pred=model.predict(x_train)
>>> mse = complex_mean_squared_error(y_train, y_pred)
>>> mse[:10]
<tf.Tensor: id=3100, shape=(10, 1, 2), dtype=float64, numpy=
array([[[1.25752882e+008, 2.08576356e+008]],

       [[6.56830262e+071, 8.71581026e+073]],

       [[2.36645002e+258, 2.05981317e+260]],

       [[1.08757145e+072, 2.62735041e+074]],

       [[9.64696113e-001, 4.07512119e+000]],

       [[1.47165401e+005, 1.11301988e+005]],

       [[2.36645002e+258, 2.05981317e+260]],

       [[3.34033795e+018, 1.28307055e+019]],

       [[3.21329232e-003, 9.95883002e+000]],

       [[2.22356245e+005, 2.32864631e+006]]])>
>>> tf.sum(mse)
Traceback (most recent call last):
  File "<pyshell#379>", line 1, in <module>
    tf.sum(mse)
AttributeError: module 'tensorflow' has no attribute 'sum'
>>> tf.reduce_sum(mse)
<tf.Tensor: id=3103, shape=(), dtype=float64, numpy=2.320616681178556e+263>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 6, in <module>
    from . import conv_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\conv_utils.py", line 9, in <module>
    from .. import backend as K
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\__init__.py", line 1, in <module>
    from .load_backend import epsilon
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\load_backend.py", line 90, in <module>
    from .tensorflow_backend import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 5, in <module>
    import tensorflow as tf
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\__init__.py", line 45, in <module>
    from tensorflow._api.v2 import compat
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\_api\v2\compat\__init__.py", line 21, in <module>
    from tensorflow._api.v2.compat import v1
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\_api\v2\compat\v1\__init__.py", line 658, in <module>
    from tensorflow.python.keras.api._v1 import keras
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\api\_v1\__init__.py", line 8, in <module>
    from tensorflow.python.keras.api._v1 import keras
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\api\_v1\keras\__init__.py", line 21, in <module>
    from tensorflow.python.keras.api._v1.keras import datasets
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\api\_v1\keras\datasets\__init__.py", line 11, in <module>
    from tensorflow.python.keras.api._v1.keras.datasets import fashion_mnist
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
  File "<frozen importlib._bootstrap_external>", line 764, in get_code
  File "<frozen importlib._bootstrap_external>", line 832, in get_data
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "<pyshell#381>", line 1, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: 0.0267 - val_loss: 0.0019
Epoch 2/50
 - 2s - loss: 0.0013 - val_loss: 8.7417e-04
Epoch 3/50
 - 2s - loss: 6.4999e-04 - val_loss: 4.9862e-04
Epoch 4/50
 - 2s - loss: 5.7666e-04 - val_loss: 4.9431e-04
Epoch 5/50
 - 1s - loss: 5.5388e-04 - val_loss: 4.5295e-04
Epoch 6/50
 - 1s - loss: 7.0222e-04 - val_loss: 4.5552e-04
Epoch 7/50
 - 1s - loss: 5.5884e-04 - val_loss: 8.4586e-04
Epoch 8/50
 - 3s - loss: 5.5586e-04 - val_loss: 4.0246e-04
Epoch 9/50
 - 2s - loss: 4.3816e-04 - val_loss: 6.4389e-04
Epoch 10/50
 - 2s - loss: 3.6285e-04 - val_loss: 3.9472e-04
Epoch 11/50
 - 2s - loss: 3.1409e-04 - val_loss: 2.4386e-04
Epoch 12/50
 - 2s - loss: 2.5077e-04 - val_loss: 3.9350e-04
Epoch 13/50
 - 2s - loss: 2.3176e-04 - val_loss: 2.2111e-04
Epoch 14/50
 - 2s - loss: 2.1398e-04 - val_loss: 1.9771e-04
Epoch 15/50
 - 1s - loss: 1.9881e-04 - val_loss: 1.8601e-04
Epoch 16/50
 - 3s - loss: 1.8852e-04 - val_loss: 9.3253e-05
Epoch 17/50
 - 3s - loss: 1.7724e-04 - val_loss: 1.1097e-04
Epoch 18/50
 - 2s - loss: 1.6575e-04 - val_loss: 6.0827e-05
Epoch 19/50
 - 2s - loss: 1.6240e-04 - val_loss: 1.3422e-04
Epoch 20/50
 - 2s - loss: 1.5287e-04 - val_loss: 1.0067e-04
Epoch 21/50
 - 2s - loss: 1.4553e-04 - val_loss: 8.1545e-05
Epoch 22/50
 - 2s - loss: 1.4067e-04 - val_loss: 2.3920e-04
Epoch 23/50
 - 2s - loss: 1.3615e-04 - val_loss: 1.0515e-04
Epoch 24/50
 - 2s - loss: 1.3179e-04 - val_loss: 1.1447e-04
Epoch 25/50
 - 2s - loss: 1.2681e-04 - val_loss: 5.1703e-05
Epoch 26/50
 - 2s - loss: 1.2499e-04 - val_loss: 9.8787e-05
Epoch 27/50
 - 1s - loss: 1.2151e-04 - val_loss: 2.2953e-05
Epoch 28/50
 - 1s - loss: 1.1685e-04 - val_loss: 1.8303e-04
Epoch 29/50
 - 1s - loss: 1.1579e-04 - val_loss: 6.6695e-05
Epoch 30/50
 - 1s - loss: 1.1208e-04 - val_loss: 2.7008e-05
Epoch 31/50
 - 1s - loss: 1.1309e-04 - val_loss: 2.8166e-05
Epoch 32/50
 - 1s - loss: 1.0804e-04 - val_loss: 5.4885e-05
Epoch 33/50
 - 1s - loss: 1.0847e-04 - val_loss: 5.8721e-05
Epoch 34/50
 - 1s - loss: 1.0611e-04 - val_loss: 1.3655e-04
Epoch 35/50
 - 1s - loss: 1.0320e-04 - val_loss: 8.9463e-05
Epoch 36/50
 - 1s - loss: 1.0381e-04 - val_loss: 1.8492e-04
Epoch 37/50
 - 1s - loss: 9.9910e-05 - val_loss: 3.8988e-05
Epoch 38/50
 - 1s - loss: 9.8859e-05 - val_loss: 4.8887e-05
Epoch 39/50
 - 1s - loss: 9.7181e-05 - val_loss: 1.5300e-04
Epoch 40/50
 - 1s - loss: 9.5417e-05 - val_loss: 7.0108e-05
Epoch 41/50
 - 1s - loss: 9.3830e-05 - val_loss: 1.3666e-04
Epoch 42/50
 - 1s - loss: 9.4711e-05 - val_loss: 2.1563e-04
Epoch 43/50
 - 3s - loss: 9.3791e-05 - val_loss: 2.0796e-05
Epoch 44/50
 - 2s - loss: 9.2726e-05 - val_loss: 1.9487e-04
Epoch 45/50
 - 1s - loss: 9.1965e-05 - val_loss: 2.2301e-05
Epoch 46/50
 - 1s - loss: 9.0664e-05 - val_loss: 1.7061e-04
Epoch 47/50
 - 1s - loss: 8.9687e-05 - val_loss: 1.4706e-04
Epoch 48/50
 - 1s - loss: 8.8846e-05 - val_loss: 7.9670e-05
Epoch 49/50
 - 1s - loss: 8.8740e-05 - val_loss: 1.6891e-04
Epoch 50/50
 - 3s - loss: 8.7198e-05 - val_loss: 2.4829e-04
>>> model.get_weights()
[array([[-0.01331576,  0.56542104, -0.75935274,  0.8002455 , -0.06247177,
         0.06508298, -0.29046947,  0.06507273],
       [-0.5635417 , -0.3612382 ,  0.3778776 , -0.52279824,  0.45901445,
         0.6572401 ,  0.12466232, -0.24668811],
       [-0.6603767 ,  0.15404549,  0.5249664 , -0.13606963, -0.55109125,
        -0.6199041 , -0.6407742 , -0.2048847 ],
       [ 0.46024385, -0.54853034,  0.17421831,  0.5188398 ,  0.1752275 ,
         0.3380757 , -0.14274634, -0.30874568]], dtype=float32), array([[ 0.50099295, -0.24070211, -0.4034095 ,  0.35339507,  0.14175552,
        -0.48724395, -0.11571226,  0.22367054],
       [ 0.3534362 ,  0.42857158,  0.7818213 , -0.41916376, -0.2434426 ,
         0.14232838, -0.19552255, -0.7279893 ],
       [ 0.09870571,  0.47291034,  0.37836173,  0.04883434, -0.73444164,
         0.01544335,  0.61103654, -0.31192642],
       [-0.06406185, -0.26750824, -0.5490075 ,  0.3557282 ,  0.36201307,
         0.00190976, -0.440283  ,  0.55002505],
       [-0.1227226 ,  0.37950635,  0.53248256, -0.00140401, -0.32384762,
        -0.1444273 , -0.3986483 ,  0.26833642],
       [-0.28902563,  0.04120145,  0.49495956, -0.27725798, -0.10271794,
        -0.25705683,  0.09291511, -0.2098174 ],
       [ 0.0943194 ,  0.12218329,  0.17390558,  0.12779029, -0.41760287,
         0.21749699,  0.611827  ,  0.1946196 ],
       [-0.09956081, -0.0683616 ,  0.33891547,  0.22418703, -0.03241273,
        -0.06246489,  0.3766021 ,  0.15568216]], dtype=float32), array([[-7.88134456e-01,  4.12056476e-01, -2.83127904e-01,
         1.10664122e-01,  3.55891466e-01,  1.73510853e-02,
         2.21182004e-01, -4.84562635e-01],
       [-2.12406926e-02,  2.63306826e-01, -1.74601050e-03,
         3.42976421e-01,  4.35788393e-01,  4.27530527e-01,
         3.84141624e-01, -3.68078321e-01],
       [ 4.24075574e-01,  1.53615206e-01, -1.24161616e-01,
        -5.00083901e-02, -9.66358185e-02,  1.20053953e-02,
        -2.52651483e-01,  3.66780281e-01],
       [-5.81374884e-01, -1.10670947e-01,  1.78123966e-01,
        -1.46930516e-01,  3.99329603e-01,  4.64476049e-01,
         5.59226610e-02, -5.72574019e-01],
       [-2.28486910e-01, -4.95840818e-01,  1.96147069e-01,
        -4.28014874e-01, -2.10899293e-01, -4.69648778e-01,
         4.29007441e-01, -2.77230382e-01],
       [ 3.10551703e-01, -4.27385598e-01,  5.07866859e-01,
        -1.75555483e-01,  5.15566647e-01,  3.59578192e-01,
         9.63497832e-02, -1.03972606e-01],
       [ 3.33397329e-01,  3.09530020e-01, -9.05226171e-02,
         4.08725917e-01,  1.97907221e-02, -2.94786781e-01,
        -2.25057900e-01,  3.47364321e-02],
       [-2.14381814e-01, -3.07467461e-01,  5.35547137e-01,
        -5.60225666e-01, -3.75930041e-01, -1.11685477e-01,
         3.75574119e-02, -1.69721067e-01],
       [-2.75632203e-01,  3.44588310e-01,  5.22960126e-01,
         4.26318765e-01,  4.14491773e-01,  5.88811278e-01,
        -2.91170448e-01,  3.56502116e-01],
       [ 1.40392315e-02, -2.09325060e-01,  5.16265512e-01,
        -5.12517035e-01, -1.14902638e-01,  1.98725775e-01,
         2.94072688e-01,  9.91902053e-02],
       [-4.43734586e-01,  2.84829456e-03,  8.49815607e-02,
        -7.73535445e-02,  4.85952944e-01, -2.25108583e-02,
        -1.44050941e-01,  3.02160442e-01],
       [-2.74494499e-01,  1.41290873e-01, -1.20855436e-01,
         1.71069413e-01,  1.87569425e-01,  1.62891462e-01,
         5.67352116e-01,  6.75968170e-01],
       [-2.99298793e-01,  2.03487590e-01,  2.02854812e-01,
         1.55088082e-01, -1.48203820e-01,  9.87917883e-05,
        -1.17206663e-01, -5.47523439e-01],
       [ 6.14089310e-01, -2.77372301e-01,  1.66675463e-01,
        -3.18717271e-01, -4.21873391e-01, -2.39083931e-01,
         4.29221153e-01,  3.07563066e-01],
       [ 1.69669576e-02, -1.37590528e-01,  1.43387413e-04,
        -5.18908560e-01, -6.36304557e-01,  1.11803047e-01,
        -1.23044848e-01, -2.25263745e-01],
       [-4.19439018e-01,  5.16245127e-01,  2.90926963e-01,
        -9.95508209e-02, -1.66663935e-03,  2.93984145e-01,
        -1.75668418e-01,  3.07720125e-01]], dtype=float32), array([[ 0.11740822],
       [-0.52708423],
       [-0.06008156],
       [-0.16395596],
       [ 0.42017326],
       [ 0.6227848 ],
       [-0.04887013],
       [ 0.7027102 ]], dtype=float32)]
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "<frozen importlib._bootstrap_external>", line 88, in _path_is_mode_type
  File "<frozen importlib._bootstrap_external>", line 82, in _path_stat
FileNotFoundError: [WinError 2] The system cannot find the file specified: 'C:\\Users\\joshm\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\tensorflow\\python\\ops\\signal\\__init__.cp36-win_amd64.pyd'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 6, in <module>
    from . import conv_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\conv_utils.py", line 9, in <module>
    from .. import backend as K
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\__init__.py", line 1, in <module>
    from .load_backend import epsilon
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\load_backend.py", line 90, in <module>
    from .tensorflow_backend import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 5, in <module>
    import tensorflow as tf
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\__init__.py", line 40, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\__init__.py", line 97, in <module>
    from tensorflow.python.ops.linalg import linalg
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\linalg\linalg.py", line 23, in <module>
    from tensorflow.python.ops.linalg import adjoint_registrations as _adjoint_registrations
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\linalg\adjoint_registrations.py", line 26, in <module>
    from tensorflow.python.ops.linalg import linear_operator_circulant
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\linalg\linear_operator_circulant.py", line 33, in <module>
    from tensorflow.python.ops.signal import fft_ops
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 951, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 894, in _find_spec
  File "<frozen importlib._bootstrap_external>", line 1157, in find_spec
  File "<frozen importlib._bootstrap_external>", line 1129, in _get_spec
  File "<frozen importlib._bootstrap_external>", line 1260, in find_spec
  File "<frozen importlib._bootstrap_external>", line 96, in _path_isfile
  File "<frozen importlib._bootstrap_external>", line 88, in _path_is_mode_type
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 112, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1', regularization=l0_1)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 47, in __init__
    **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\legacy\interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\layers\core.py", line 873, in __init__
    super(Dense, self).__init__(**kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 147, in __init__
    raise TypeError('Keyword argument not understood:', kwarg)
TypeError: ('Keyword argument not understood:', 'regularization')
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 527, in _apply_op_helper
    preferred_dtype=default_dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1036, in _TensorTensorConversionFunction
    (dtype.name, t.dtype.name, str(t)))
ValueError: Tensor conversion requested dtype float64 for Tensor with dtype float32: 'Tensor("power_dense_1/weight_regularizer/norm/Squeeze:0", shape=(), dtype=float32)'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 118, in <module>
    model.compile(loss=complex_mean_squared_error,optimizer="rmsprop", metrics=[])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 229, in compile
    self.total_loss = self._prepare_total_loss(masks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 713, in _prepare_total_loss
    total_loss += loss_tensor
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 884, in binary_op_wrapper
    return func(x, y, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 396, in add
    "Add", x=x, y=y, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 563, in _apply_op_helper
    inferred_from[input_arg.type_attr]))
TypeError: Input 'y' of 'Add' Op has type float32 that does not match type float64 of argument 'x'.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
 - 2s - loss: nan - val_loss: nan
Epoch 4/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 129, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 130, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 130, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 130, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 130, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: nan - val_loss: nan
Epoch 2/50
 - 3s - loss: nan - val_loss: nan
Epoch 3/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 130, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: nan - val_loss: nan
Epoch 2/50
 - 3s - loss: nan - val_loss: nan
Epoch 3/50
 - 2s - loss: nan - val_loss: nan
Epoch 4/50
 - 2s - loss: nan - val_loss: nan
Epoch 5/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 130, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 130, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> np.max(x_train)
-1.0001588094558702
>>> y_pred=model.predict(x_train)
>>> np.max(y_pred)
nan
>>> y_pred[:10]
array([[[nan, nan]],

       [[nan, nan]],

       [[nan, nan]],

       [[nan, nan]],

       [[nan, nan]],

       [[nan, nan]],

       [[nan, nan]],

       [[nan, nan]],

       [[nan, nan]],

       [[nan, nan]]])
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> 
>>> y_pred=model.predict(x_train)
>>> np.max(y_pred)
4.73065356083417
>>> np.min(y_pred)
-4.713239535928111
>>> y_pred = tf.convert_to_tensor(y_pred)
>>> y_true = tf.stack((y_train,tf.zeros_like(y_train)),axis=2)
>>> y_pred.shape
TensorShape([8000, 1, 2])
>>> y_train.shape
(8000, 1)
>>> y_true.shape
TensorShape([8000, 1, 2])
>>> y_true = tf.cast(y_true, y_pred.dtype)
>>> mse = tf.cast(tf.square(y_pred - y_true),tf.float64)
>>> mse.shape
TensorShape([8000, 1, 2])
>>> np.max(mse)
23.922027905391758
>>> l0_1(model.get_weights())
Traceback (most recent call last):
  File "<pyshell#400>", line 1, in <module>
    l0_1(model.get_weights())
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 88, in l0_1
    return tf.cast(tf.norm(weight_matrix,ord=0.1),tf.float64)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\linalg_ops.py", line 493, in norm_v2
    name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\linalg_ops.py", line 593, in norm
    tensor = ops.convert_to_tensor(tensor)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1100, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 254, in _constant_impl
    t = convert_to_eager_tensor(value, ctx, dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 115, in convert_to_eager_tensor
    return ops.EagerTensor(value, handle, device, dtype)
ValueError: Can't convert non-rectangular Python sequence to Tensor.
>>> l0_1(model.get_weights()[0])
<tf.Tensor: id=3615, shape=(), dtype=float64, numpy=349183022202880.0>
>>> l0_1(model.get_weights()[1])
<tf.Tensor: id=3634, shape=(), dtype=float64, numpy=2.626735512139858e+17>
>>> l0_1(model.get_weights()[2])
<tf.Tensor: id=3653, shape=(), dtype=float64, numpy=2.2534793396869084e+20>
>>> l0_1(model.get_weights()[3])
<tf.Tensor: id=3672, shape=(), dtype=float64, numpy=169443360.0>
>>> l0_1(model.get_weights()[4])
Traceback (most recent call last):
  File "<pyshell#405>", line 1, in <module>
    l0_1(model.get_weights()[4])
IndexError: list index out of range
>>> model.get_weights()[1].shape
(8, 8)
>>> model.get_weights()[2].shape
(16, 8)
>>> model.get_weights()[1]
array([[ 0.21245801,  0.42809385, -0.45930767, -0.00228989, -0.3667676 ,
         0.2668839 , -0.21681786,  0.54212624],
       [-0.04831523,  0.08640063,  0.03323525, -0.34261698,  0.56736606,
         0.06383729,  0.36403692,  0.12651086],
       [-0.53055674, -0.23774812, -0.39876866,  0.19582236,  0.5369218 ,
        -0.19823575,  0.40438944, -0.02675831],
       [ 0.3315876 , -0.486763  , -0.3437504 , -0.22211522,  0.1275419 ,
         0.30243897, -0.16974366,  0.47580713],
       [ 0.33959186,  0.4424817 ,  0.40508705,  0.34415495,  0.13690805,
         0.5092104 ,  0.38962144,  0.46601492],
       [ 0.07791036,  0.6053358 , -0.02549237, -0.31071094,  0.47108597,
        -0.40865993,  0.48266262, -0.08362061],
       [ 0.31130183,  0.4675935 , -0.11865249,  0.11701173,  0.01474363,
        -0.14175424,  0.17568576,  0.45843822],
       [-0.3183052 ,  0.07769692, -0.22013324, -0.40859935,  0.16534287,
        -0.48018405,  0.56296223, -0.33791298]], dtype=float32)
>>> np.pow(abs(model.get_weights()[1]),0.1)
Traceback (most recent call last):
  File "<pyshell#409>", line 1, in <module>
    np.pow(abs(model.get_weights()[1]),0.1)
AttributeError: module 'numpy' has no attribute 'pow'
>>> tf.pow(abs(model.get_weights()[1]),0.1)
<tf.Tensor: id=3716, shape=(8, 8), dtype=float32, numpy=
array([[0.8564999 , 0.9186581 , 0.9251462 , 0.5444794 , 0.90456355,
        0.8762585 , 0.8582415 , 0.940611  ],
       [0.73859847, 0.78280157, 0.71147543, 0.8984231 , 0.94490105,
        0.75946444, 0.9038878 , 0.81322885],
       [0.9385841 , 0.8661871 , 0.91216224, 0.8495447 , 0.93970406,
        0.85058594, 0.9134399 , 0.696219  ],
       [0.89548814, 0.93053293, 0.89871985, 0.8603157 , 0.8138892 ,
        0.8872863 , 0.8374895 , 0.928417  ],
       [0.89762664, 0.9216999 , 0.91359735, 0.8988255 , 0.8196773 ,
        0.9347376 , 0.91004795, 0.9264884 ],
       [0.7747463 , 0.9510419 , 0.69285285, 0.8896837 , 0.92749166,
        0.9144    , 0.9297461 , 0.7802456 ],
       [0.88985276, 0.92680174, 0.80803037, 0.806906  , 0.65593445,
        0.82253355, 0.840376  , 0.9249709 ],
       [0.89183474, 0.7745338 , 0.8595449 , 0.9143864 , 0.83529246,
        0.9292675 , 0.94416505, 0.89718187]], dtype=float32)>
>>> tf.reduce_sum(tf.pow(abs(model.get_weights()[1]),0.1))
<tf.Tensor: id=3730, shape=(), dtype=float32, numpy=55.200325>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 6s - loss: 1.7737 - val_loss: 0.2022
Epoch 2/50
 - 3s - loss: 0.2010 - val_loss: 0.1970
Epoch 3/50
 - 2s - loss: 0.1969 - val_loss: 0.1950
Epoch 4/50
 - 2s - loss: 0.1943 - val_loss: 0.1922
Epoch 5/50
 - 3s - loss: 0.1913 - val_loss: 0.1887
Epoch 6/50
 - 2s - loss: 0.1878 - val_loss: 0.1857
Epoch 7/50
 - 2s - loss: 0.1840 - val_loss: 0.1821
Epoch 8/50
 - 1s - loss: 0.1798 - val_loss: 0.1779
Epoch 9/50
 - 1s - loss: 0.1754 - val_loss: 0.1734
Epoch 10/50
 - 1s - loss: 0.1723 - val_loss: 0.1694
Epoch 11/50
 - 1s - loss: 0.1701 - val_loss: 0.1691
Epoch 12/50
 - 1s - loss: 0.1676 - val_loss: 0.1659
Epoch 13/50
 - 1s - loss: 0.1648 - val_loss: 0.1638
Epoch 14/50
 - 1s - loss: 0.1627 - val_loss: 0.1604
Epoch 15/50
 - 3s - loss: 0.1603 - val_loss: 0.1588
Epoch 16/50
 - 3s - loss: 0.1582 - val_loss: 0.1578
Epoch 17/50
 - 2s - loss: 0.1565 - val_loss: 0.1562
Epoch 18/50
 - 2s - loss: 0.1554 - val_loss: 0.1539
Epoch 19/50
 - 2s - loss: 0.1537 - val_loss: 0.1514
Epoch 20/50
 - 2s - loss: 0.1515 - val_loss: 0.1513
Epoch 21/50
 - 2s - loss: 0.1502 - val_loss: 0.1507
Epoch 22/50
Traceback (most recent call last):
  File "<pyshell#412>", line 1, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 5, in <module>
    from . import io_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\io_utils.py", line 14, in <module>
    import h5py
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\h5py\__init__.py", line 59, in <module>
    from ._hl.files import (
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\h5py\_hl\files.py", line 25, in <module>
    from .group import Group
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\h5py\_hl\group.py", line 26, in <module>
    from . import dataset
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\h5py\_hl\dataset.py", line 30, in <module>
    from . import selections as sel
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
  File "<frozen importlib._bootstrap_external>", line 758, in get_code
  File "<frozen importlib._bootstrap_external>", line 842, in path_stats
  File "<frozen importlib._bootstrap_external>", line 82, in _path_stat
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 6, in <module>
    from . import conv_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\conv_utils.py", line 9, in <module>
    from .. import backend as K
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\__init__.py", line 1, in <module>
    from .load_backend import epsilon
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\load_backend.py", line 90, in <module>
    from .tensorflow_backend import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 5, in <module>
    import tensorflow as tf
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\__init__.py", line 40, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\__init__.py", line 73, in <module>
    from tensorflow.python.ops.standard_ops import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\standard_ops.py", line 61, in <module>
    from tensorflow.python.ops.gradients import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gradients.py", line 26, in <module>
    from tensorflow.python.ops.gradients_impl import gradients
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gradients_impl.py", line 30, in <module>
    from tensorflow.python.ops import linalg_grad  # pylint: disable=unused-import
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\linalg_grad.py", line 34, in <module>
    from tensorflow.python.ops import linalg_ops
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\linalg_ops.py", line 29, in <module>
    from tensorflow.python.ops import map_fn
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 951, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 894, in _find_spec
  File "<frozen importlib._bootstrap_external>", line 1157, in find_spec
  File "<frozen importlib._bootstrap_external>", line 1129, in _get_spec
  File "<frozen importlib._bootstrap_external>", line 1241, in find_spec
  File "<frozen importlib._bootstrap_external>", line 82, in _path_stat
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 130, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_squared_error/Sum (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_2720]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 136, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_squared_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_2724]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
 - 3s - loss: nan - val_loss: nan
Epoch 3/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 136, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 192, in fit_loop
    callbacks._call_batch_hook('train', 'begin', batch_index, batch_logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 88, in _call_batch_hook
    delta_t_median = np.median(self._delta_ts[hook_name])
  File "<__array_function__ internals>", line 6, in median
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3502, in median
    overwrite_input=overwrite_input)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3410, in _ureduce
    r = func(a, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3535, in _median
    part = partition(a, kth, axis=axis)
  File "<__array_function__ internals>", line 6, in partition
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\fromnumeric.py", line 745, in partition
    a.partition(kth, axis=axis, kind=kind, order=order)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 136, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: 0.2201 - val_loss: 0.1950
Epoch 2/50
 - 2s - loss: 0.1904 - val_loss: 0.1847
Epoch 3/50
 - 1s - loss: 0.1813 - val_loss: 0.1794
Epoch 4/50
 - 1s - loss: 0.1763 - val_loss: 0.1740
Epoch 5/50
 - 1s - loss: 0.1716 - val_loss: 0.1702
Epoch 6/50
 - 1s - loss: 0.1691 - val_loss: 0.1681
Epoch 7/50
 - 3s - loss: 0.1666 - val_loss: 0.1653
Epoch 8/50
 - 2s - loss: 0.1624 - val_loss: 0.1599
Epoch 9/50
 - 2s - loss: 0.1578 - val_loss: 0.1565
Epoch 10/50
 - 2s - loss: 0.1561 - val_loss: 0.1557
Epoch 11/50
 - 2s - loss: 0.1552 - val_loss: 0.1547
Epoch 12/50
 - 2s - loss: 0.1530 - val_loss: 0.1506
Epoch 13/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 136, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 136, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_squared_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_2724]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/10
 - 5s - loss: 111.9326 - val_loss: 0.2182
Epoch 2/10
 - 2s - loss: 0.2120 - val_loss: 0.2067
Epoch 3/10
 - 2s - loss: 0.2013 - val_loss: 0.1974
Epoch 4/10
 - 2s - loss: 0.1981 - val_loss: 0.1963
Epoch 5/10
 - 2s - loss: 0.1952 - val_loss: 0.1927
Epoch 6/10
 - 2s - loss: 0.1919 - val_loss: 0.1901
Epoch 7/10
 - 1s - loss: 0.1899 - val_loss: 0.1886
Epoch 8/10
 - 1s - loss: 0.1872 - val_loss: 0.1852
Epoch 9/10
 - 1s - loss: 0.1841 - val_loss: 0.1826
Epoch 10/10
 - 1s - loss: 0.1807 - val_loss: 0.1804
>>> model.get_weights()
[array([[-1.1488854e-01, -2.2780347e-01,  6.8314008e-02, -4.0792567e-01,
         5.0595605e-01, -2.9883483e-01, -5.9460918e-04,  3.6585030e-01],
       [-2.6318058e-01, -1.1676089e-01,  2.5425997e-01,  4.9674752e-01,
        -7.1274745e-01, -3.7569460e-01, -3.2776970e-01, -4.1617301e-01],
       [-2.5063062e-01,  6.4244121e-01, -6.2260264e-01,  2.0724521e-03,
        -5.6976162e-02, -1.8191051e-01, -5.1410431e-01,  5.2252847e-01],
       [-4.4857132e-01,  6.2282634e-01,  3.6963925e-01, -4.4800270e-01,
         2.9255219e-02,  3.7634748e-01,  6.1110282e-01,  1.3931404e-02]],
      dtype=float32), array([[ 2.5942063e-01, -2.6869330e-01, -3.3918628e-01,  4.7146699e-01,
        -6.2518328e-01,  2.9375369e-03, -1.5158519e-01, -6.1341882e-01],
       [-4.4013572e-01,  3.1071084e-03, -1.5088886e-02, -4.2016143e-01,
         4.2051535e-02,  1.9838042e-03, -8.1004776e-02,  6.0475057e-01],
       [-1.0137749e-01,  2.6821056e-01,  4.8979267e-01, -1.7764776e-03,
         2.7678909e-03, -1.1516813e-03, -3.7482390e-04, -1.0834470e-01],
       [ 4.0240425e-01,  2.0332046e-02,  1.5197827e-01, -6.1921316e-01,
        -4.3185094e-01,  2.3233355e-03,  1.8780078e-01,  2.6917704e-03],
       [-6.6291523e-01, -2.9657000e-01, -4.1449617e-04,  4.5946085e-01,
         1.1012551e-01,  4.0922791e-02,  3.6007136e-01,  2.5134030e-01],
       [ 2.6994562e-01, -1.8563916e-01,  5.4662798e-02, -3.9329210e-01,
         2.2208700e-03,  2.5048212e-05, -5.7935995e-01,  3.1097338e-01],
       [-3.6974469e-01,  3.9561373e-01, -2.2718147e-03,  3.7917724e-01,
         2.3693970e-01,  1.9741017e-01, -2.0974779e-01, -4.1038112e-04],
       [ 1.7376612e-01,  4.7310561e-01, -5.9606582e-01, -8.3875000e-02,
        -2.9675728e-03,  9.1404666e-04,  2.6031157e-02, -3.5383070e-01]],
      dtype=float32), array([[-1.0361555e-03,  1.8896448e-04,  4.3559569e-01, -8.8637992e-04,
        -4.5376408e-01,  2.4639529e-03, -3.0596943e-03, -7.6172814e-02],
       [-3.3150849e-01, -2.4449572e-04,  4.8479307e-01, -1.9036046e-01,
        -2.3433782e-01,  2.3140316e-01,  1.9389865e-01, -3.1468316e-03],
       [-1.6387455e-01,  4.9215223e-04, -1.4996321e-03, -3.5937357e-01,
        -3.5084662e-01,  2.8032082e-01, -8.2212754e-02,  2.3906205e-03],
       [ 3.6387698e-04, -1.7323121e-01, -2.8713611e-03, -4.3699276e-01,
         3.2113829e-01,  3.5440069e-01,  4.7038889e-01, -2.0609917e-01],
       [-2.3378269e-01, -2.8627256e-01, -2.3989458e-04,  1.0081697e-03,
        -1.2075633e-01, -4.3839195e-01, -1.6785160e-03,  9.4450049e-02],
       [ 1.1501380e-01,  7.7054508e-02,  7.1069354e-04, -2.4637408e-03,
         1.9464999e-01,  3.1028917e-01,  3.5668740e-01, -1.1220190e-02],
       [ 2.6292723e-01,  3.3405492e-01,  4.3236125e-01, -2.3770840e-03,
        -1.9537159e-03, -5.1514160e-02,  2.7373409e-01,  1.3461241e-01],
       [ 1.4963001e-01, -8.7901280e-04, -1.6536238e-03,  2.8115276e-03,
        -2.0761701e-01, -4.5160285e-01, -4.4200531e-01, -4.8481897e-04],
       [-1.0039978e-01, -4.1105419e-01, -1.7389397e-01, -3.4304467e-01,
         7.9748477e-04,  3.0004740e-01,  2.4906184e-01,  2.3824909e-01],
       [-3.0626038e-01, -9.1962342e-04,  3.5971177e-01,  2.4617827e-01,
        -4.1274214e-01, -2.9418364e-01,  3.0526519e-03,  9.5518753e-02],
       [ 1.6903126e-03,  8.5711218e-02, -2.2155562e-01,  1.2239087e-03,
         2.5532078e-03, -5.2328545e-01, -2.4500275e-03, -1.8390751e-03],
       [ 5.5925723e-04, -1.6642518e-03,  1.7475671e-01, -2.7254359e-03,
         4.1656715e-01,  3.1262620e-03,  2.4632711e-03,  2.0492531e-01],
       [-3.0529931e-01, -3.7977588e-01, -2.9479504e-01, -2.8723148e-03,
        -3.8735127e-01,  1.5199903e-01, -6.9873914e-04, -4.3691468e-01],
       [-2.0449921e-03,  1.9555509e-03,  3.9569891e-04,  2.1868641e-03,
         1.8708367e-04, -5.6584150e-02, -2.2285185e-03, -9.9953517e-02],
       [-2.9990569e-01, -3.1300017e-01, -3.2082456e-01, -2.0818153e-01,
         5.0374007e-01,  3.1188816e-01,  4.1694924e-01,  3.1560340e-03],
       [ 2.0311809e-01, -3.1215617e-01,  8.3152637e-02,  3.5788512e-01,
        -4.8099023e-01,  7.9933293e-02,  2.5816753e-03, -2.7632827e-03]],
      dtype=float32), array([[ 0.749871  ],
       [-0.23615475],
       [-0.4454478 ],
       [-0.41646922],
       [-0.32179707],
       [-0.7396913 ],
       [ 0.7034327 ],
       [ 0.6601943 ]], dtype=float32)]
>>> y_pred = model.predict(x_train)
>>> y_pred = model.predict(x_test)
>>> complex_mean_squared_error(y_test, y_pred)
<tf.Tensor: id=19693, shape=(), dtype=float64, numpy=0.0014599714721215053>
>>> l0_1(model.get_weights()[1])
<tf.Tensor: id=19716, shape=(), dtype=float64, numpy=0.04969314956665039>
>>> l0_1(model.get_weights()[2])
<tf.Tensor: id=19739, shape=(), dtype=float64, numpy=0.09440180206298829>
>>> l0_1(model.get_weights()[3])
<tf.Tensor: id=19762, shape=(), dtype=float64, numpy=0.007463545799255371>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 2s - loss: 8.5736 - val_loss: 0.2247
Epoch 2/50
 - 1s - loss: 0.2208 - val_loss: 0.2060
Epoch 3/50
 - 1s - loss: 0.1935 - val_loss: 0.1875
Epoch 4/50
 - 1s - loss: 0.1831 - val_loss: 0.1785
Epoch 5/50
 - 1s - loss: 0.1766 - val_loss: 0.1740
Epoch 6/50
 - 1s - loss: 0.1715 - val_loss: 0.1695
Epoch 7/50
 - 1s - loss: 0.1674 - val_loss: 0.1664
Epoch 8/50
 - 1s - loss: 0.1642 - val_loss: 0.1626
Epoch 9/50
 - 1s - loss: 0.1602 - val_loss: 0.1573
Epoch 10/50
 - 1s - loss: 0.1560 - val_loss: 0.1534
Epoch 11/50
 - 1s - loss: 0.1540 - val_loss: 0.1524
Epoch 12/50
 - 1s - loss: 0.1541 - val_loss: 0.1532
Epoch 13/50
 - 1s - loss: 0.1528 - val_loss: 0.1512
Epoch 14/50
 - 1s - loss: 0.1508 - val_loss: 0.1494
Epoch 15/50
 - 1s - loss: 0.1487 - val_loss: 0.1475
Epoch 16/50
 - 1s - loss: 0.1470 - val_loss: 0.1467
Epoch 17/50
 - 1s - loss: 0.1459 - val_loss: 0.1462
Epoch 18/50
 - 1s - loss: 0.1452 - val_loss: 0.1438
Epoch 19/50
 - 1s - loss: 0.1439 - val_loss: 0.1427
Epoch 20/50
 - 1s - loss: 0.1423 - val_loss: 0.1423
Epoch 21/50
 - 1s - loss: 0.1409 - val_loss: 0.1409
Epoch 22/50
 - 1s - loss: 0.1398 - val_loss: 0.1387
Epoch 23/50
 - 1s - loss: 0.1388 - val_loss: 0.1395
Epoch 24/50
 - 1s - loss: 0.1388 - val_loss: 0.1385
Epoch 25/50
 - 1s - loss: 0.1387 - val_loss: 0.1392
Epoch 26/50
 - 2s - loss: 0.1385 - val_loss: 0.1388
Epoch 27/50
 - 2s - loss: 0.1380 - val_loss: 0.1385
Epoch 28/50
 - 2s - loss: 0.1379 - val_loss: 0.1376
Epoch 29/50
 - 2s - loss: 0.1376 - val_loss: 0.1384
Epoch 30/50
 - 2s - loss: 0.1374 - val_loss: 0.1377
Epoch 31/50
 - 2s - loss: 0.1369 - val_loss: 0.1364
Epoch 32/50
 - 2s - loss: 0.1370 - val_loss: 0.1363
Epoch 33/50
 - 2s - loss: 0.1365 - val_loss: 0.1355
Epoch 34/50
 - 2s - loss: 0.1361 - val_loss: 0.1357
Epoch 35/50
 - 1s - loss: 0.1559 - val_loss: 0.1414
Epoch 36/50
 - 2s - loss: 0.1414 - val_loss: 0.1379
Epoch 37/50
 - 2s - loss: 0.1374 - val_loss: 0.1366
Epoch 38/50
 - 2s - loss: 0.1354 - val_loss: 0.1342
Epoch 39/50
 - 2s - loss: 0.1339 - val_loss: 0.1338
Epoch 40/50
 - 2s - loss: 0.1337 - val_loss: 0.1335
Epoch 41/50
 - 2s - loss: 0.1340 - val_loss: 0.1335
Epoch 42/50
 - 2s - loss: 0.1332 - val_loss: 0.1353
Epoch 43/50
 - 2s - loss: 0.1325 - val_loss: 0.1326
Epoch 44/50
 - 2s - loss: 0.1320 - val_loss: 0.1332
Epoch 45/50
 - 2s - loss: 0.1315 - val_loss: 0.1318
Epoch 46/50
 - 2s - loss: 0.1309 - val_loss: 0.1298
Epoch 47/50
 - 2s - loss: 0.1303 - val_loss: 0.1304
Epoch 48/50
 - 2s - loss: 0.1298 - val_loss: 0.1293
Epoch 49/50
 - 2s - loss: 0.1293 - val_loss: 0.1284
Epoch 50/50
 - 2s - loss: 0.1284 - val_loss: 0.1278
>>> y_pred = model.predict(x_test)
>>> complex_mean_squared_error(y_test,y_pred)
<tf.Tensor: id=81535, shape=(), dtype=float64, numpy=0.0003965193060932395>
>>> y_pred[:10]
array([[[ 4.29186363e-01,  1.47875070e-02]],

       [[ 3.13937721e-01, -7.24745159e-03]],

       [[ 2.41368158e-01, -3.70372124e-04]],

       [[ 2.07472552e-01,  7.53842148e-03]],

       [[ 5.42049794e-02,  1.09363959e-02]],

       [[ 1.52645021e-01,  6.03242644e-03]],

       [[ 7.38934916e-01,  2.54596973e-02]],

       [[ 1.07903768e-02,  1.37691333e-02]],

       [[-8.14327156e-02,  3.45183284e-03]],

       [[ 1.78492666e-01,  4.30818142e-03]]])
>>> y_test[:10]
array([[ 0.46990047],
       [ 0.31543626],
       [ 0.24999835],
       [ 0.22711854],
       [ 0.07001089],
       [ 0.16558139],
       [ 0.77479398],
       [ 0.03077125],
       [-0.06716691],
       [ 0.19058633]])
>>> y_pred[:10,0,0]
array([ 0.42918636,  0.31393772,  0.24136816,  0.20747255,  0.05420498,
        0.15264502,  0.73893492,  0.01079038, -0.08143272,  0.17849267])
>>> y_pred[:10,:,0]
array([[ 0.42918636],
       [ 0.31393772],
       [ 0.24136816],
       [ 0.20747255],
       [ 0.05420498],
       [ 0.15264502],
       [ 0.73893492],
       [ 0.01079038],
       [-0.08143272],
       [ 0.17849267]])
>>> model.get_weights()
[array([[-1.5581243e-03, -2.2005726e-04, -7.0628210e-04,  2.3462810e-04,
         9.0109429e-04, -6.5818816e-01, -1.0658139e+00,  1.7663977e-03],
       [ 6.1685801e-05,  1.0606640e-03, -2.4202326e-03, -1.4392434e-03,
        -2.8176124e-03,  6.1299878e-01, -2.2349319e-01, -1.3444412e-03],
       [ 2.5581834e-03, -1.9830512e-03,  8.9146895e-05,  1.6359043e-03,
         1.8316644e-01,  3.2188499e-01, -7.2064175e-04,  5.3718901e-04],
       [ 3.1285223e-03, -4.1249068e-04,  7.1508408e-01, -2.4283703e-03,
        -1.1553437e-03,  8.6218566e-02,  1.7639822e-03,  2.0864913e-03]],
      dtype=float32), array([[ 2.6384550e-03,  1.7721907e-03,  7.5165991e-04,  1.9662166e-03,
        -2.5907466e-03,  2.0370851e-03,  2.2178146e-03, -2.4982898e-03],
       [ 2.4556534e-03,  2.9310237e-03,  1.2678891e-01, -2.0977121e-04,
        -6.1613636e-04,  1.3845945e-03,  3.0799275e-03, -2.6218451e-03],
       [-1.9585525e-03, -7.9329178e-04,  8.3692282e-02, -2.9165347e-03,
        -2.2678745e-03, -2.2516754e-03, -1.7146735e-03,  3.0199024e-03],
       [ 4.4748260e-04,  7.1608997e-04,  3.8108742e-04, -4.9515511e-06,
         3.0432083e-04,  7.1856356e-04, -2.9612710e-03, -3.0704206e-03],
       [ 2.2139419e-03, -1.2160139e-03,  1.9543313e-03,  2.5668424e-03,
         3.0916306e-04, -1.4990219e-01, -4.7878790e-04,  6.4645871e-04],
       [ 2.4313104e-01, -1.6875251e-03, -4.5721582e-01,  2.4231609e-03,
         5.5987376e-04,  2.5766381e-04, -1.7942212e-03, -1.2366099e-03],
       [ 1.4959355e-03, -1.4498413e-01,  3.1386591e-03, -1.1757892e-03,
        -2.6434020e-03,  1.8142833e-03,  2.7536598e-01, -1.2413774e-03],
       [-1.8886051e-03, -1.4638245e-03, -2.8227263e-03, -9.3381194e-04,
         1.3083610e-04, -1.2446756e-03,  1.1842082e-03, -8.1303588e-04]],
      dtype=float32), array([[ 3.13331070e-03,  2.33239541e-03,  1.31742435e-03,
        -2.74544070e-03, -7.38867209e-04,  2.15061754e-03,
        -1.19072990e-03, -2.72160536e-03],
       [-5.61164110e-04,  1.28402223e-03,  1.93855225e-03,
         4.02862788e-04,  2.95532425e-03,  1.94066868e-03,
        -2.28585140e-03,  7.58390757e-04],
       [-4.73053777e-04,  1.79064635e-03,  2.78289220e-03,
        -2.97374302e-03, -1.32263126e-03,  3.11539765e-03,
        -3.07237991e-04, -5.14604035e-04],
       [ 5.95241901e-04, -9.91691137e-04,  2.40434567e-03,
         5.90058393e-04, -1.24152610e-03, -1.43433688e-03,
         2.75291502e-03,  2.89720669e-03],
       [ 1.16728700e-03, -2.90914765e-03,  1.97754032e-03,
         2.30614445e-03, -1.45325367e-03,  7.80323637e-04,
        -9.90657834e-04, -1.02254038e-04],
       [ 8.07930133e-04, -3.13285948e-03,  3.09248641e-03,
         2.72212666e-03, -3.14901536e-03, -1.46659859e-03,
         2.99286563e-03, -7.53673608e-04],
       [ 3.76113474e-01,  4.15966497e-05,  1.52792304e-03,
        -8.83866509e-04, -4.98981623e-04, -6.99848053e-04,
        -2.03136820e-03, -1.34288182e-03],
       [-1.13678339e-03, -1.31624390e-03,  1.05150801e-03,
        -3.09933699e-03, -2.61404947e-03,  1.11583853e-03,
         2.96818768e-03,  3.00810277e-03],
       [ 1.27793569e-03, -1.45096681e-04,  2.49472400e-03,
         3.00989463e-03,  7.19573232e-04,  1.70540437e-03,
         2.87599978e-04, -4.77373481e-01],
       [-2.92685931e-03,  1.25396135e-03, -1.72809546e-03,
        -1.51900004e-03,  2.47759256e-03,  1.91510189e-05,
        -9.95382130e-01,  1.50444382e-03],
       [ 2.81399256e-03, -1.83096010e-04,  1.71968190e-04,
         3.77142365e-04,  2.38662283e-03,  5.32018603e-04,
        -1.15730043e-03, -3.18868071e-01],
       [-1.32063846e-03, -8.00090143e-04,  1.82697584e-03,
        -2.98331887e-03,  3.08125629e-03, -2.00292445e-03,
         5.89778356e-05, -1.08415086e-04],
       [ 2.95885024e-03,  4.96206747e-04,  9.93147143e-04,
        -2.44109938e-03,  1.00573432e-03, -2.19332613e-03,
        -4.43347672e-04,  1.62073446e-03],
       [ 2.63497210e-03,  2.68906960e-03, -8.02140683e-04,
        -1.09801826e-03, -1.45985186e-03,  1.79555058e-03,
        -1.55097502e-03, -1.59639101e-02],
       [-4.80277551e-04,  2.14656349e-03, -2.32290593e-03,
         2.38994416e-03,  2.25725205e-04, -1.75415585e-03,
         7.16659124e-05,  1.16024713e-03],
       [-3.57331941e-04, -1.20570813e-03, -5.66271425e-04,
         2.17392645e-03, -2.27171346e-03,  2.05679424e-03,
        -1.01359026e-03,  1.81802956e-04]], dtype=float32), array([[-0.20848303],
       [ 0.19965208],
       [-0.16526143],
       [ 0.00154754],
       [ 0.00164517],
       [ 0.00097688],
       [ 0.18702047],
       [ 0.3584134 ]], dtype=float32)]
>>> (10**-4)**0.1
0.39810717055349726
>>> (10**-5)**0.1
0.31622776601683794
>>> (10**1)**0.1
1.2589254117941673
>>> (10**2)**0.1
1.5848931924611136
>>> tf.relu(10)
Traceback (most recent call last):
  File "<pyshell#431>", line 1, in <module>
    tf.relu(10)
AttributeError: module 'tensorflow' has no attribute 'relu'
>>> tf.nn.relu([-3,-2,-1,0,1,2,3])
<tf.Tensor: id=81546, shape=(7,), dtype=int32, numpy=array([0, 0, 0, 0, 1, 2, 3])>
>>> tf.nn.relu([-3,-2,-1,0,-4,-5,-6]+3)
Traceback (most recent call last):
  File "<pyshell#433>", line 1, in <module>
    tf.nn.relu([-3,-2,-1,0,-4,-5,-6]+3)
TypeError: can only concatenate list (not "int") to list
>>> tf.nn.relu(tf.add_bias([-3,-2,-1,0,-4,-5,-6],3))
Traceback (most recent call last):
  File "<pyshell#434>", line 1, in <module>
    tf.nn.relu(tf.add_bias([-3,-2,-1,0,-4,-5,-6],3))
AttributeError: module 'tensorflow' has no attribute 'add_bias'
>>> tf.nn.relu(tf.nn.add_bias([-3,-2,-1,0,-4,-5,-6],3))
Traceback (most recent call last):
  File "<pyshell#435>", line 1, in <module>
    tf.nn.relu(tf.nn.add_bias([-3,-2,-1,0,-4,-5,-6],3))
AttributeError: module 'tensorflow._api.v2.nn' has no attribute 'add_bias'
>>> tf.nn.relu(tf.add([-3,-2,-1,0,-4,-5,-6],3))
<tf.Tensor: id=81551, shape=(7,), dtype=int32, numpy=array([0, 1, 2, 3, 0, 0, 0])>
>>> tf.nn.relu(tf.add([0,-1,-2,-3,-4,-5,-6],3))
<tf.Tensor: id=81556, shape=(7,), dtype=int32, numpy=array([3, 2, 1, 0, 0, 0, 0])>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 122, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1', kernel_regularizer=llog)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 463, in __call__
    self.build(unpack_singleton(input_shapes))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 84, in build
    constraint=self.kernel_constraint)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 285, in add_weight
    self.add_loss(regularizer(weight))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 91, in llog
    values = tf.nn.relu(tf.add(tf.log(weight_matrix),20))
AttributeError: module 'tensorflow' has no attribute 'log'
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
 - 2s - loss: nan - val_loss: nan
Epoch 4/50
 - 2s - loss: nan - val_loss: nan
Epoch 5/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 140, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> model.get_weights()
[array([[-0.00277438, -0.06236936, -0.49718094,  0.34376833, -0.28934488,
        -0.15677884, -0.00289993, -0.26657984],
       [-0.53230256, -0.01542029, -0.00366202, -0.00307201,  0.5566105 ,
        -0.04940063, -0.00078359, -0.30213454],
       [ 0.11688936,  0.1985516 , -0.17009482,  0.31164992, -0.01431426,
         0.58395714, -0.00302385,  0.4801867 ],
       [ 0.530371  ,  0.31696033, -0.48782685,  0.6937841 ,  0.69793653,
         0.09663476, -0.54316825, -0.18207048]], dtype=float32), array([[-0.5191957 , -0.3299219 , -0.00115436, -0.14255345,  0.28461924,
        -0.19662541,  0.27261204, -0.45070252],
       [-0.36015442, -0.45354816,  0.4387559 , -0.02699657, -0.45590264,
        -0.00384286, -0.362955  , -0.26789558],
       [ 0.13502128, -0.17788833, -0.1789468 , -0.00312269,  0.29840124,
         0.010918  ,  0.4730288 , -0.5006877 ],
       [ 0.39770666, -0.03302552, -0.03494576, -0.1598774 , -0.48319954,
        -0.00269448, -0.00221349, -0.08545959],
       [-0.23677409, -0.09914093, -0.11478335,  0.3839218 , -0.00331725,
        -0.17179942, -0.63512546, -0.09602676],
       [-0.27374026, -0.02530807, -0.11760537,  0.3436573 , -0.22555159,
         0.4705923 , -0.01720664,  0.06161155],
       [ 0.01533795, -0.00178882, -0.00081802,  0.2884702 , -0.2116053 ,
        -0.00290012, -0.00302141, -0.17623933],
       [-0.60373855, -0.38686028,  0.22344668, -0.53483707, -0.08414046,
        -0.15700762, -0.50945014, -0.21372543]], dtype=float32), array([[-0.21419495, -0.3295129 , -0.35543305, -0.44694382,  0.09967746,
        -0.07516612, -0.3159717 ,  0.3709041 ],
       [-0.28165555, -0.33094832, -0.03227245, -0.50708914, -0.33268932,
        -0.16832024, -0.00262027, -0.24030922],
       [-0.00862392, -0.38461646, -0.21546407, -0.00853096, -0.00141016,
         0.2115016 , -0.08072761, -0.03529052],
       [ 0.271965  , -0.42847177, -0.22346948,  0.29610476,  0.0585582 ,
        -0.36227575, -0.16118634,  0.5209735 ],
       [-0.06744552,  0.16061038, -0.3514472 , -0.35559633, -0.32070154,
        -0.06520867, -0.00299852, -0.32094985],
       [-0.01567636, -0.01170637, -0.4408655 , -0.20558529, -0.38022587,
        -0.00600044, -0.01446838, -0.06814727],
       [-0.3777398 ,  0.24880421, -0.00886511,  0.24872035, -0.2933327 ,
        -0.20648623, -0.0026406 , -0.10795184],
       [ 0.3886774 , -0.00287749, -0.27233478,  0.17844708, -0.3094739 ,
        -0.49045932, -0.08314944, -0.00267362],
       [-0.00071115,  0.338014  , -0.5562813 , -0.11936624, -0.1000517 ,
        -0.347261  , -0.00204603,  0.167645  ],
       [ 0.06746073, -0.4354252 , -0.32730737, -0.00367777, -0.39356214,
        -0.46002823, -0.10578044,  0.35965398],
       [-0.01833275, -0.00575434, -0.23801932, -0.00507909,  0.42157584,
         0.06415766, -0.43625277, -0.0068204 ],
       [ 0.40269443, -0.3363276 , -0.29311472,  0.09731597,  0.21653546,
        -0.06334773,  0.30926085,  0.3154481 ],
       [ 0.26318124, -0.36184984, -0.05063983, -0.00247221, -0.13738486,
        -0.40326473, -0.0030339 , -0.24503517],
       [ 0.33656183,  0.3132836 , -0.25573587,  0.32121453, -0.14195913,
         0.26606745, -0.00264379, -0.00285773],
       [-0.18269959, -0.00767396, -0.04833564,  0.09698401, -0.23628314,
        -0.3223966 , -0.13706528,  0.43529683],
       [-0.00285232, -0.00301448, -0.01431717, -0.17154953, -0.01329611,
        -0.06836551, -0.00226802,  0.2173318 ]], dtype=float32), array([[-0.71582556],
       [-0.41620633],
       [-0.2828189 ],
       [-0.34248966],
       [-0.7122096 ],
       [-0.50541675],
       [-0.24040453],
       [-0.25010693]], dtype=float32)]
>>> y_pred=model.predict(x_train)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 141, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 210, in fit_loop
    verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 449, in test_loop
    batch_outs = f(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> y_pred=model.predict(x_train)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: 8.7748 - val_loss: 8.6381
Epoch 2/50
 - 3s - loss: 8.5601 - val_loss: 8.4792
Epoch 3/50
 - 2s - loss: 8.3856 - val_loss: 8.3112
Epoch 4/50
 - 1s - loss: 8.2438 - val_loss: 8.1643
Epoch 5/50
 - 1s - loss: 8.1141 - val_loss: 8.0874
Epoch 6/50
 - 1s - loss: 8.0618 - val_loss: 8.0357
Epoch 7/50
 - 1s - loss: 8.0008 - val_loss: 7.9512
Epoch 8/50
 - 1s - loss: 7.9642 - val_loss: 7.9666
Epoch 9/50
 - 1s - loss: 7.9566 - val_loss: 7.9610
Epoch 10/50
 - 1s - loss: 7.9410 - val_loss: 7.9344
Epoch 11/50
 - 1s - loss: 7.9280 - val_loss: 7.8930
Epoch 12/50
 - 1s - loss: 7.9111 - val_loss: 7.9259
Epoch 13/50
 - 3s - loss: 7.9054 - val_loss: 7.8726
Epoch 14/50
 - 3s - loss: 7.8968 - val_loss: 7.8882
Epoch 15/50
 - 3s - loss: 7.8895 - val_loss: 7.9080
Epoch 16/50
 - 2s - loss: 7.8845 - val_loss: 7.8891
Epoch 17/50
 - 1s - loss: 7.8793 - val_loss: 7.8888
Epoch 18/50
 - 1s - loss: 7.9278 - val_loss: 8.0566
Epoch 19/50
 - 1s - loss: 7.9873 - val_loss: 7.9289
Epoch 20/50
 - 3s - loss: 7.9105 - val_loss: 7.8839
Epoch 21/50
 - 2s - loss: 7.8568 - val_loss: 7.8438
Epoch 22/50
 - 2s - loss: 7.8282 - val_loss: 7.8271
Epoch 23/50
 - 2s - loss: 7.8194 - val_loss: 7.8326
Epoch 24/50
 - 2s - loss: 7.8157 - val_loss: 7.8204
Epoch 25/50
 - 2s - loss: 7.8206 - val_loss: 7.7961
Epoch 26/50
 - 2s - loss: 7.8752 - val_loss: 7.8092
Epoch 27/50
 - 1s - loss: 7.9209 - val_loss: 7.8601
Epoch 28/50
 - 2s - loss: 7.8563 - val_loss: 7.8624
Epoch 29/50
 - 1s - loss: 7.8532 - val_loss: 7.8591
Epoch 30/50
 - 1s - loss: 7.8453 - val_loss: 7.8526
Epoch 31/50
 - 1s - loss: 7.8375 - val_loss: 7.8293
Epoch 32/50
 - 1s - loss: 7.8404 - val_loss: 7.8510
Epoch 33/50
 - 1s - loss: 7.8364 - val_loss: 7.8278
Epoch 34/50
 - 1s - loss: 7.8371 - val_loss: 7.8367
Epoch 35/50
 - 1s - loss: 7.8402 - val_loss: 7.8478
Epoch 36/50
 - 1s - loss: 7.8361 - val_loss: 7.8563
Epoch 37/50
 - 3s - loss: 7.8386 - val_loss: 7.8332
Epoch 38/50
 - 2s - loss: 7.8399 - val_loss: 7.8326
Epoch 39/50
 - 2s - loss: 7.8319 - val_loss: 7.8431
Epoch 40/50
 - 2s - loss: 7.8307 - val_loss: 7.8412
Epoch 41/50
 - 1s - loss: 7.8322 - val_loss: 7.8072
Epoch 42/50
 - 1s - loss: 7.8328 - val_loss: 7.8362
Epoch 43/50
 - 1s - loss: 7.8313 - val_loss: 7.8434
Epoch 44/50
 - 1s - loss: 7.8304 - val_loss: 7.8400
Epoch 45/50
 - 1s - loss: 7.8321 - val_loss: 7.8412
Epoch 46/50
 - 1s - loss: 7.8308 - val_loss: 7.8446
Epoch 47/50
 - 1s - loss: 7.8286 - val_loss: 7.8325
Epoch 48/50
 - 1s - loss: 7.8328 - val_loss: 7.8219
Epoch 49/50
 - 1s - loss: 7.8292 - val_loss: 7.8368
Epoch 50/50
 - 1s - loss: 7.8236 - val_loss: 7.8244
>>> model.get_weights()
[array([[ 2.0210741e-03,  2.1061269e-03,  1.1746883e-03, -2.8149800e-03,
        -2.8591978e-03,  7.6505874e-04, -1.3340131e-04, -2.0896015e-03],
       [-1.4446820e-03,  2.8823612e-03,  3.7639285e-05,  1.4440890e-04,
         1.3515573e-03,  1.5858602e-04, -2.4015629e-03,  2.6597558e-03],
       [-1.5625600e-03, -2.3045046e-03, -4.0455736e-04, -3.7264156e-01,
         4.6358534e-04, -2.9006817e-03, -1.2392048e-03, -9.0894941e-04],
       [ 9.6488622e-04, -2.6208682e-03, -1.5599339e-04,  4.0879717e-01,
         2.1999599e-03, -3.0037325e-03, -1.3837570e-03,  3.0478803e-03]],
      dtype=float32), array([[ 1.5802163e-03, -5.5204669e-04, -1.9034900e-03, -7.6200702e-04,
        -2.3567090e-03,  2.0239882e-03,  9.1688277e-04, -3.2237720e-01],
       [-2.6726318e-03,  2.1658933e-03, -2.8057920e-03, -1.8120679e-03,
        -3.0575884e-03,  1.5550333e-03, -1.0439762e-03,  1.5324975e-03],
       [-1.4096280e-03, -2.3309120e-03,  5.2910054e-04,  4.1124603e-04,
         2.5351516e-03, -1.6104230e-03, -1.0259743e-03, -2.0190906e-03],
       [ 2.9545410e-03, -2.4046751e-03, -2.4246804e-03,  5.3399116e-02,
         8.8234915e-04,  1.1488707e-03, -6.1493769e-04,  2.9660955e-01],
       [ 3.9301915e-04, -6.7834661e-04, -2.5516278e-03, -2.3902822e-03,
        -2.2365807e-03,  3.1524808e-03, -2.9000691e-03,  2.8887491e-03],
       [-9.6515706e-04,  7.2330656e-04, -6.8812631e-04, -1.7639183e-03,
         2.2409775e-03,  1.1805113e-04,  9.1050559e-04, -3.1361086e-03],
       [-1.8655033e-03, -2.8503984e-03, -1.2214871e-03, -2.8717921e-03,
        -1.9274090e-03,  1.8736599e-03,  2.4010784e-03, -3.0186749e-03],
       [ 1.5737006e-04,  4.6943250e-04,  1.3867068e-03,  2.6183791e-04,
        -2.4374824e-03, -7.3653454e-04,  2.6411661e-03, -2.0092144e-03]],
      dtype=float32), array([[ 2.4322206e-03, -2.8975136e-03,  4.0718514e-04, -4.4718577e-04,
        -2.4684563e-03,  2.0950809e-03, -7.8146218e-04, -2.7032814e-04],
       [ 1.4307054e-03,  1.7919729e-03, -2.6844440e-03,  1.9266437e-03,
         6.5248786e-04,  2.1514430e-04, -2.5420747e-04,  1.7842074e-04],
       [ 3.1207136e-03, -1.8033360e-03,  1.6054485e-03,  1.4429849e-03,
         7.7159243e-04, -1.0804300e-03,  8.9305616e-04, -2.7991123e-03],
       [-1.2906416e-03, -2.0831276e-03,  2.6182826e-03,  2.7979396e-03,
         2.1859310e-03,  3.5206077e-04,  3.6587386e-04, -1.7719613e-03],
       [ 1.0710048e-03, -1.3253819e-03,  1.7723596e-03, -1.8337837e-03,
         2.3980136e-03, -1.8959951e-03, -8.9251192e-04,  1.4790863e-03],
       [ 7.1478932e-04, -2.6584012e-03,  3.1520072e-03,  2.2257245e-03,
        -2.7090672e-03, -2.8539724e-03, -2.4288800e-04,  2.6317982e-03],
       [ 3.1431748e-03,  1.9185633e-03,  2.5367034e-03, -2.5711504e-03,
         3.3085234e-04, -1.6052101e-03,  2.0490917e-03,  1.8647920e-03],
       [-1.7758785e-03,  3.6720268e-04, -3.1617354e-03, -2.6198446e-03,
        -2.8270246e-03, -1.9830039e-03, -5.9088535e-04,  4.4466375e-04],
       [ 2.8134317e-03, -1.1702126e-05,  4.1733071e-04, -2.1474776e-03,
        -8.9862512e-04, -2.3491795e-03,  2.2272870e-03,  2.1742948e-03],
       [ 2.4689620e-04, -2.8821058e-03,  1.9051484e-05,  1.2083562e-03,
        -3.0885306e-03, -5.8662292e-04, -2.2942282e-03, -1.5179330e-04],
       [ 2.1688249e-03, -1.9564610e-03, -1.6748194e-03,  1.1427223e-03,
        -2.6462171e-03, -2.8069220e-03, -2.6077686e-03, -2.1273009e-03],
       [ 1.9536328e-03, -2.7489688e-03, -2.8483339e-03,  2.6583909e-03,
        -1.1267903e-03, -3.0308173e-03, -3.6853188e-01,  2.5407984e-03],
       [-2.1925890e-03, -6.0944597e-04,  2.8394361e-03, -2.6863969e-03,
         2.3056536e-03,  2.1152876e-03, -1.5247574e-03, -6.8717590e-04],
       [ 2.0312264e-03, -1.7841632e-04, -3.1092069e-03,  2.9649739e-03,
        -1.6366050e-03, -8.2118809e-02, -2.4017725e-02, -6.4735196e-04],
       [-2.2284724e-03,  1.5758777e-03,  2.7833572e-03,  7.7486545e-04,
         7.2616560e-05,  1.4904914e-03,  7.0760393e-04,  2.9782280e-03],
       [-2.6564950e-01, -1.6533548e-03,  2.9643413e-03, -1.8163712e-03,
         3.0959733e-03,  1.9061925e-03, -1.5500732e-01,  3.0024087e-03]],
      dtype=float32), array([[ 0.63995147],
       [-0.00203291],
       [-0.00205892],
       [-0.00313937],
       [-0.0027806 ],
       [ 0.6218773 ],
       [-0.36132905],
       [ 0.00220753]], dtype=float32)]
>>> weights = model.get_weights()
>>> for r in range(len(weights[i])):
	for c in range(len(weights[i][0])):
		if abs(weights[i][r][c])<0.01:
			weights[i][r][c]=0

			
Traceback (most recent call last):
  File "<pyshell#448>", line 1, in <module>
    for r in range(len(weights[i])):
NameError: name 'i' is not defined
>>> i=0
>>> for r in range(len(weights[i])):
	for c in range(len(weights[i][0])):
		if abs(weights[i][r][c])<0.01:
			weights[i][r][c]=0

			
>>> weights[0]
array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        , -0.37264156,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.40879717,  0.        ,
         0.        ,  0.        ,  0.        ]], dtype=float32)
>>> i=1
>>> for r in range(len(weights[i])):
	for c in range(len(weights[i][0])):
		if abs(weights[i][r][c])<0.01:
			weights[i][r][c]=0

			
>>> i=2
>>> for r in range(len(weights[i])):
	for c in range(len(weights[i][0])):
		if abs(weights[i][r][c])<0.01:
			weights[i][r][c]=0

			
>>> i=3
>>> for r in range(len(weights[i])):
	for c in range(len(weights[i][0])):
		if abs(weights[i][r][c])<0.01:
			weights[i][r][c]=0

			
>>> weights
[array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        , -0.37264156,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.40879717,  0.        ,
         0.        ,  0.        ,  0.        ]], dtype=float32), array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        , -0.3223772 ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.05339912,  0.        ,
         0.        ,  0.        ,  0.29660955],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ]], dtype=float32), array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        , -0.36853188,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        -0.08211881, -0.02401773,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [-0.2656495 ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        , -0.15500732,  0.        ]], dtype=float32), array([[ 0.63995147],
       [ 0.        ],
       [ 0.        ],
       [ 0.        ],
       [ 0.        ],
       [ 0.6218773 ],
       [-0.36132905],
       [ 0.        ]], dtype=float32)]
>>> weightsoriginal = model.get_weights()
>>> model.set_weights(weights)
>>> y_pred=model.predict(x_test)
>>> complex_mean_squared_error(y_test,y_pred)
<tf.Tensor: id=81251, shape=(), dtype=float64, numpy=7.169758558095898e+85>
>>> model.set_weights(weightsoriginal)
>>> y_pred=model.predict(x_test)
>>> complex_mean_squared_error(y_test,y_pred)
<tf.Tensor: id=81455, shape=(), dtype=float64, numpy=0.016044121583991364>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: 13.5410 - val_loss: 12.9962
Epoch 2/50
 - 1s - loss: 12.7936 - val_loss: 12.6087
Epoch 3/50
 - 1s - loss: 12.4069 - val_loss: 12.7009
Epoch 4/50
 - 1s - loss: 12.2619 - val_loss: 12.2138
Epoch 5/50
 - 1s - loss: 12.2115 - val_loss: 11.6801
Epoch 6/50
 - 1s - loss: 12.1407 - val_loss: 11.5590
Epoch 7/50
 - 1s - loss: 12.0529 - val_loss: 12.0166
Epoch 8/50
 - 1s - loss: 12.1073 - val_loss: 11.9963
Epoch 9/50
 - 1s - loss: 12.0868 - val_loss: 12.4924
Epoch 10/50
 - 1s - loss: 12.0875 - val_loss: 11.9794
Epoch 11/50
 - 1s - loss: 12.0320 - val_loss: 11.9193
Epoch 12/50
 - 1s - loss: 11.9717 - val_loss: 12.0226
Epoch 13/50
 - 1s - loss: 11.9130 - val_loss: 12.3670
Epoch 14/50
 - 1s - loss: 11.9830 - val_loss: 11.4546
Epoch 15/50
 - 1s - loss: 11.8987 - val_loss: 11.8427
Epoch 16/50
 - 1s - loss: 11.6733 - val_loss: 11.7520
Epoch 17/50
 - 1s - loss: 11.8711 - val_loss: 11.2876
Epoch 18/50
 - 1s - loss: 11.6738 - val_loss: 11.8523
Epoch 19/50
 - 1s - loss: 11.6519 - val_loss: 11.1730
Epoch 20/50
 - 1s - loss: 11.4002 - val_loss: 11.5824
Epoch 21/50
 - 1s - loss: 11.5746 - val_loss: 11.3779
Epoch 22/50
 - 1s - loss: 11.4695 - val_loss: 10.8285
Epoch 23/50
 - 1s - loss: 11.4758 - val_loss: 12.0511
Epoch 24/50
 - 1s - loss: 11.4503 - val_loss: 11.5218
Epoch 25/50
 - 1s - loss: 11.4101 - val_loss: 11.1460
Epoch 26/50
 - 1s - loss: 11.3586 - val_loss: 11.5475
Epoch 27/50
 - 1s - loss: 11.4609 - val_loss: 11.5007
Epoch 28/50
 - 1s - loss: 11.3328 - val_loss: 11.5073
Epoch 29/50
 - 1s - loss: 11.3796 - val_loss: 12.2509
Epoch 30/50
 - 1s - loss: 11.4713 - val_loss: 11.5973
Epoch 31/50
 - 1s - loss: 11.3174 - val_loss: 11.0891
Epoch 32/50
 - 1s - loss: 11.2013 - val_loss: 11.2108
Epoch 33/50
 - 1s - loss: 11.1416 - val_loss: 11.6454
Epoch 34/50
 - 1s - loss: 11.2646 - val_loss: 11.5573
Epoch 35/50
 - 1s - loss: 11.2396 - val_loss: 11.3974
Epoch 36/50
 - 1s - loss: 11.1309 - val_loss: 11.0462
Epoch 37/50
 - 1s - loss: 11.1248 - val_loss: 11.0530
Epoch 38/50
 - 1s - loss: 11.1456 - val_loss: 11.5828
Epoch 39/50
 - 1s - loss: 11.1496 - val_loss: 11.8156
Epoch 40/50
 - 1s - loss: 11.1709 - val_loss: 11.2961
Epoch 41/50
 - 1s - loss: 11.3082 - val_loss: 11.8572
Epoch 42/50
 - 1s - loss: 11.0798 - val_loss: 10.5551
Epoch 43/50
 - 1s - loss: 11.2292 - val_loss: 11.0794
Epoch 44/50
 - 1s - loss: 11.0375 - val_loss: 10.7488
Epoch 45/50
 - 1s - loss: 11.1182 - val_loss: 11.2621
Epoch 46/50
 - 1s - loss: 11.1036 - val_loss: 10.8672
Epoch 47/50
 - 1s - loss: 11.0259 - val_loss: 10.9111
Epoch 48/50
 - 1s - loss: 10.9748 - val_loss: 10.8785
Epoch 49/50
 - 1s - loss: 11.0400 - val_loss: 10.9489
Epoch 50/50
 - 1s - loss: 11.0957 - val_loss: 11.1935
>>> model.get_weights
<bound method Network.get_weights of <keras.engine.training.Model object at 0x0000026C0D7F78D0>>
>>> model.get_weights()
[array([[ 0.05478142, -0.2727785 ,  0.4718936 ,  0.55663943, -0.19298108,
         0.17487942, -0.5034138 ,  0.48989818],
       [-0.43251663,  0.49077603, -0.39160523,  0.15469883,  0.6720864 ,
        -0.4365745 ,  0.60772914,  0.0941958 ],
       [-0.4538517 ,  0.3833847 ,  0.13990204, -0.13917105, -0.5261207 ,
        -0.0456752 , -0.36849162,  0.48977283],
       [ 0.12159707, -0.1365645 , -0.12149064,  0.38432056,  0.14569193,
        -0.39654574,  0.3963853 , -0.40072984]], dtype=float32), array([[-3.16652238e-01, -3.51768792e-01,  2.56467909e-01,
         3.68922681e-01,  4.02334511e-01,  4.57478911e-01,
        -4.31024656e-03,  1.53306825e-02],
       [ 4.00140546e-02,  5.22410572e-01,  5.60234487e-01,
        -1.53066471e-01,  1.51485890e-01,  5.99850297e-01,
         2.48793140e-01,  3.95089388e-01],
       [-1.90944076e-01, -1.71516225e-01,  1.55310944e-01,
         3.18889618e-01,  3.80099744e-01, -4.74018157e-01,
         2.70098805e-01,  3.35757673e-01],
       [ 3.88018608e-01, -1.98623493e-01,  5.48761725e-01,
        -9.10655945e-04,  2.86855161e-01, -2.53499672e-03,
        -3.85339320e-01, -2.99574994e-02],
       [ 4.59937006e-01,  2.26068437e-01,  3.78396034e-01,
        -4.15486425e-01, -4.07451302e-01, -4.92132962e-01,
         4.07064527e-01, -6.07022405e-01],
       [-4.16816592e-01,  1.27357855e-01,  3.79929364e-01,
         1.50325716e-01,  4.23354059e-01,  1.62048608e-01,
         3.63736898e-01, -4.18986268e-02],
       [-1.35393813e-03,  4.58457947e-01,  1.35838538e-01,
        -4.71782237e-01,  2.24643841e-01, -3.47253343e-04,
        -5.19647419e-01, -6.71369070e-03],
       [-5.86326361e-01, -1.58349931e-01,  5.82377732e-01,
        -5.43845236e-01,  4.61353928e-01, -7.47615620e-02,
         1.02695405e-01,  4.05354053e-01]], dtype=float32), array([[-0.14132342,  0.18132068,  0.23158047, -0.21355706,  0.15448342,
        -0.0016469 ,  0.16990441, -0.00109413],
       [ 0.00309806,  0.12777974, -0.57486665,  0.0032699 , -0.00313516,
        -0.16293032,  0.21872559,  0.38983083],
       [-0.58224696, -0.55487967, -0.38264886, -0.54560053, -0.39382267,
         0.08006286,  0.03964731, -0.589743  ],
       [-0.00249477, -0.00102264,  0.49029323, -0.29106456,  0.63953376,
        -0.26107875,  0.37612432,  0.4984329 ],
       [ 0.22953047, -0.3117559 ,  0.38058984, -0.10058095,  0.16810012,
         0.29513192, -0.46471396, -0.00546291],
       [-0.49619916, -0.23683174, -0.32548773,  0.08191403, -0.38959068,
        -0.45163143,  0.27242443,  0.00452459],
       [-0.25743154,  0.4472048 , -0.14259109,  0.24056324,  0.00352757,
        -0.26992905, -0.10108966,  0.2757958 ],
       [-0.0024216 , -0.23785633, -0.22468896,  0.07808259, -0.17488456,
        -0.11353501, -0.3562606 ,  0.40877688],
       [-0.00285325, -0.41654915, -0.35402885, -0.2831921 , -0.00140583,
        -0.1610227 ,  0.26976725,  0.43406507],
       [ 0.62187284, -0.33339408, -0.39984688,  0.07317051, -0.08800843,
         0.35133067,  0.16021335,  0.6870856 ],
       [-0.50089294, -0.31887558,  0.0684742 ,  0.20001394, -0.3421349 ,
         0.18933652,  0.21357572,  0.08880734],
       [ 0.42853302, -0.20985118,  0.13471508,  0.16215159, -0.17072177,
        -0.1348381 , -0.4441779 ,  0.0133095 ],
       [ 0.00255674, -0.0524997 ,  0.53254944,  0.48252183, -0.34916934,
         0.00447139, -0.48261085, -0.00257221],
       [ 0.13556807,  0.2776967 ,  0.23784763,  0.37286532,  0.39908764,
         0.22094317,  0.18539684, -0.05752822],
       [ 0.13621168,  0.23667495,  0.27191824,  0.18145555, -0.16444406,
         0.3183537 , -0.38167337,  0.11700886],
       [ 0.3297972 ,  0.28301525, -0.26196465,  0.21504022,  0.0008375 ,
        -0.3852947 ,  0.10804546,  0.1944978 ]], dtype=float32), array([[-0.24282882],
       [ 0.27247882],
       [-0.21225296],
       [-0.36527607],
       [ 0.3438398 ],
       [ 0.32940254],
       [ 0.5957497 ],
       [-0.49025965]], dtype=float32)]
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 136, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_squared_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_2370]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/200
 - 3s - loss: 52.1679 - val_loss: 51.0766
Epoch 2/200
 - 1s - loss: 50.4209 - val_loss: 50.1989
Epoch 3/200
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 136, in <module>
    model.fit(x_train, y_train,epochs=200,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 201, in fit_loop
    callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 88, in _call_batch_hook
    delta_t_median = np.median(self._delta_ts[hook_name])
  File "<__array_function__ internals>", line 6, in median
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3502, in median
    overwrite_input=overwrite_input)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3410, in _ureduce
    r = func(a, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3535, in _median
    part = partition(a, kth, axis=axis)
  File "<__array_function__ internals>", line 6, in partition
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\fromnumeric.py", line 745, in partition
    a.partition(kth, axis=axis, kind=kind, order=order)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/200
 - 2s - loss: 51.8686 - val_loss: 50.9528
Epoch 2/200
 - 1s - loss: 50.8473 - val_loss: 50.4274
Epoch 3/200
 - 1s - loss: 50.1468 - val_loss: 49.9340
Epoch 4/200
 - 1s - loss: 49.6925 - val_loss: 49.5941
Epoch 5/200
 - 1s - loss: 49.3113 - val_loss: 49.3281
Epoch 6/200
 - 1s - loss: 49.0433 - val_loss: 48.9269
Epoch 7/200
 - 1s - loss: 48.7649 - val_loss: 48.5315
Epoch 8/200
 - 1s - loss: 48.4808 - val_loss: 48.2765
Epoch 9/200
 - 1s - loss: 48.3052 - val_loss: 48.1783
Epoch 10/200
 - 2s - loss: 48.1402 - val_loss: 47.9223
Epoch 11/200
 - 1s - loss: 47.7346 - val_loss: 47.6640
Epoch 12/200
 - 2s - loss: 47.5255 - val_loss: 47.4614
Epoch 13/200
 - 2s - loss: 47.3985 - val_loss: 47.3910
Epoch 14/200
 - 2s - loss: 47.3095 - val_loss: 47.3105
Epoch 15/200
 - 2s - loss: 47.2436 - val_loss: 47.0260
Epoch 16/200
 - 1s - loss: 47.1848 - val_loss: 47.2402
Epoch 17/200
 - 2s - loss: 47.0671 - val_loss: 46.9095
Epoch 18/200
 - 2s - loss: 46.9771 - val_loss: 46.9660
Epoch 19/200
 - 2s - loss: 46.9212 - val_loss: 46.9540
Epoch 20/200
 - 2s - loss: 46.7252 - val_loss: 46.5887
Epoch 21/200
 - 2s - loss: 46.5119 - val_loss: 46.5757
Epoch 22/200
 - 2s - loss: 46.4235 - val_loss: 46.2686
Epoch 23/200
 - 1s - loss: 46.2684 - val_loss: 46.1271
Epoch 24/200
 - 2s - loss: 46.1102 - val_loss: 46.3078
Epoch 25/200
 - 2s - loss: 46.0099 - val_loss: 45.8555
Epoch 26/200
 - 1s - loss: 46.0349 - val_loss: 46.2913
Epoch 27/200
 - 1s - loss: 46.0153 - val_loss: 46.0329
Epoch 28/200
 - 2s - loss: 45.9147 - val_loss: 45.9114
Epoch 29/200
 - 2s - loss: 45.8413 - val_loss: 45.8304
Epoch 30/200
 - 1s - loss: 45.8572 - val_loss: 45.9924
Epoch 31/200
 - 2s - loss: 45.7143 - val_loss: 45.7201
Epoch 32/200
 - 2s - loss: 45.6942 - val_loss: 45.5575
Epoch 33/200
 - 2s - loss: 45.5509 - val_loss: 45.7072
Epoch 34/200
 - 2s - loss: 45.5373 - val_loss: 45.4144
Epoch 35/200
 - 2s - loss: 45.4483 - val_loss: 45.3509
Epoch 36/200
 - 2s - loss: 45.3635 - val_loss: 45.2093
Epoch 37/200
 - 2s - loss: 45.3589 - val_loss: 45.3881
Epoch 38/200
 - 1s - loss: 45.3430 - val_loss: 45.4333
Epoch 39/200
 - 2s - loss: 45.2524 - val_loss: 45.6551
Epoch 40/200
 - 2s - loss: 45.2636 - val_loss: 45.0655
Epoch 41/200
 - 2s - loss: 45.2223 - val_loss: 44.9810
Epoch 42/200
 - 2s - loss: 45.1611 - val_loss: 44.9793
Epoch 43/200
 - 2s - loss: 45.0860 - val_loss: 45.1046
Epoch 44/200
 - 2s - loss: 45.1057 - val_loss: 44.8737
Epoch 45/200
 - 2s - loss: 45.0621 - val_loss: 44.8807
Epoch 46/200
 - 2s - loss: 45.0214 - val_loss: 45.1219
Epoch 47/200
 - 2s - loss: 44.9680 - val_loss: 44.9725
Epoch 48/200
 - 1s - loss: 44.9851 - val_loss: 44.9100
Epoch 49/200
 - 2s - loss: 44.9278 - val_loss: 44.9434
Epoch 50/200
 - 2s - loss: 44.9485 - val_loss: 44.6904
Epoch 51/200
 - 2s - loss: 44.8071 - val_loss: 44.8278
Epoch 52/200
 - 2s - loss: 44.8191 - val_loss: 44.8641
Epoch 53/200
 - 2s - loss: 44.7605 - val_loss: 44.8847
Epoch 54/200
 - 2s - loss: 44.6767 - val_loss: 44.8068
Epoch 55/200
 - 2s - loss: 44.5915 - val_loss: 44.3772
Epoch 56/200
 - 2s - loss: 44.5020 - val_loss: 44.6787
Epoch 57/200
 - 1s - loss: 44.4386 - val_loss: 44.3919
Epoch 58/200
 - 2s - loss: 44.4224 - val_loss: 44.3380
Epoch 59/200
 - 2s - loss: 44.3291 - val_loss: 44.2210
Epoch 60/200
 - 2s - loss: 44.2540 - val_loss: 44.3105
Epoch 61/200
 - 2s - loss: 44.1175 - val_loss: 44.2419
Epoch 62/200
 - 2s - loss: 44.0687 - val_loss: 43.8162
Epoch 63/200
 - 2s - loss: 43.9709 - val_loss: 44.0025
Epoch 64/200
 - 2s - loss: 43.9680 - val_loss: 44.0648
Epoch 65/200
 - 2s - loss: 43.7208 - val_loss: 43.7253
Epoch 66/200
 - 2s - loss: 43.7160 - val_loss: 43.7209
Epoch 67/200
 - 2s - loss: 43.5967 - val_loss: 43.7880
Epoch 68/200
 - 2s - loss: 43.5069 - val_loss: 43.6856
Epoch 69/200
 - 1s - loss: 43.4756 - val_loss: 43.4610
Epoch 70/200
 - 2s - loss: 43.3927 - val_loss: 43.4817
Epoch 71/200
 - 2s - loss: 43.3636 - val_loss: 43.1514
Epoch 72/200
 - 1s - loss: 43.2491 - val_loss: 43.1804
Epoch 73/200
 - 2s - loss: 43.1754 - val_loss: 43.3713
Epoch 74/200
 - 2s - loss: 43.1838 - val_loss: 43.5003
Epoch 75/200
 - 2s - loss: 43.1390 - val_loss: 43.3526
Epoch 76/200
 - 2s - loss: 43.0039 - val_loss: 42.9744
Epoch 77/200
 - 1s - loss: 43.0016 - val_loss: 42.9078
Epoch 78/200
 - 2s - loss: 42.8373 - val_loss: 42.7056
Epoch 79/200
 - 1s - loss: 42.6992 - val_loss: 42.5785
Epoch 80/200
 - 2s - loss: 42.6067 - val_loss: 42.4857
Epoch 81/200
 - 2s - loss: 42.4873 - val_loss: 42.5490
Epoch 82/200
 - 2s - loss: 42.4309 - val_loss: 42.2590
Epoch 83/200
 - 2s - loss: 42.3809 - val_loss: 42.3918
Epoch 84/200
 - 2s - loss: 42.2797 - val_loss: 42.6092
Epoch 85/200
 - 2s - loss: 42.2413 - val_loss: 42.0555
Epoch 86/200
 - 2s - loss: 42.1799 - val_loss: 42.2547
Epoch 87/200
 - 1s - loss: 42.1038 - val_loss: 41.9666
Epoch 88/200
 - 1s - loss: 42.1001 - val_loss: 42.3696
Epoch 89/200
 - 2s - loss: 42.0234 - val_loss: 41.9582
Epoch 90/200
 - 2s - loss: 42.0409 - val_loss: 41.7771
Epoch 91/200
 - 1s - loss: 42.0252 - val_loss: 41.8194
Epoch 92/200
 - 1s - loss: 41.9258 - val_loss: 41.9644
Epoch 93/200
 - 2s - loss: 41.9188 - val_loss: 41.8262
Epoch 94/200
 - 2s - loss: 41.8328 - val_loss: 41.9243
Epoch 95/200
 - 1s - loss: 41.6844 - val_loss: 41.6287
Epoch 96/200
 - 2s - loss: 41.6850 - val_loss: 41.8926
Epoch 97/200
 - 1s - loss: 41.7324 - val_loss: 41.9877
Epoch 98/200
 - 2s - loss: 41.7298 - val_loss: 41.5928
Epoch 99/200
 - 2s - loss: 41.6486 - val_loss: 42.3262
Epoch 100/200
 - 2s - loss: 41.6447 - val_loss: 41.7768
Epoch 101/200
 - 2s - loss: 41.6146 - val_loss: 41.5663
Epoch 102/200
 - 2s - loss: 41.5733 - val_loss: 41.7670
Epoch 103/200
 - 1s - loss: 41.5990 - val_loss: 41.8956
Epoch 104/200
 - 2s - loss: 41.4874 - val_loss: 41.5691
Epoch 105/200
 - 2s - loss: 41.4640 - val_loss: 41.4249
Epoch 106/200
 - 2s - loss: 41.3737 - val_loss: 41.4452
Epoch 107/200
 - 1s - loss: 41.3244 - val_loss: 41.4014
Epoch 108/200
 - 2s - loss: 41.2551 - val_loss: 41.3904
Epoch 109/200
 - 2s - loss: 41.2083 - val_loss: 41.1456
Epoch 110/200
 - 1s - loss: 41.1528 - val_loss: 41.2118
Epoch 111/200
 - 2s - loss: 41.0699 - val_loss: 41.2769
Epoch 112/200
 - 2s - loss: 41.1009 - val_loss: 40.9885
Epoch 113/200
 - 2s - loss: 41.0217 - val_loss: 40.9105
Epoch 114/200
 - 2s - loss: 40.9669 - val_loss: 40.4331
Epoch 115/200
 - 2s - loss: 40.8951 - val_loss: 40.9883
Epoch 116/200
 - 1s - loss: 40.8876 - val_loss: 40.9214
Epoch 117/200
 - 2s - loss: 40.8049 - val_loss: 41.0477
Epoch 118/200
 - 2s - loss: 40.8157 - val_loss: 40.8264
Epoch 119/200
 - 2s - loss: 40.6494 - val_loss: 40.4227
Epoch 120/200
 - 2s - loss: 40.5649 - val_loss: 40.4190
Epoch 121/200
 - 2s - loss: 40.5557 - val_loss: 40.3649
Epoch 122/200
 - 2s - loss: 40.5456 - val_loss: 40.3281
Epoch 123/200
 - 1s - loss: 40.1982 - val_loss: 39.9192
Epoch 124/200
 - 1s - loss: 39.9062 - val_loss: 39.9828
Epoch 125/200
 - 2s - loss: 39.7191 - val_loss: 39.4961
Epoch 126/200
 - 2s - loss: 39.5433 - val_loss: 39.1169
Epoch 127/200
 - 1s - loss: 39.4235 - val_loss: 39.1624
Epoch 128/200
 - 2s - loss: 39.2977 - val_loss: 39.1326
Epoch 129/200
 - 1s - loss: 39.1178 - val_loss: 39.2067
Epoch 130/200
 - 2s - loss: 38.9426 - val_loss: 38.5152
Epoch 131/200
 - 2s - loss: 38.6130 - val_loss: 38.8807
Epoch 132/200
 - 1s - loss: 38.2967 - val_loss: 38.5312
Epoch 133/200
 - 1s - loss: 37.9276 - val_loss: 37.6164
Epoch 134/200
 - 1s - loss: 37.4752 - val_loss: 37.4871
Epoch 135/200
 - 1s - loss: 37.1454 - val_loss: 37.1153
Epoch 136/200
 - 1s - loss: 36.7195 - val_loss: 36.6624
Epoch 137/200
 - 1s - loss: 36.5107 - val_loss: 36.3617
Epoch 138/200
 - 1s - loss: 36.2393 - val_loss: 36.3206
Epoch 139/200
 - 1s - loss: 36.2331 - val_loss: 36.0873
Epoch 140/200
 - 1s - loss: 36.1652 - val_loss: 35.9421
Epoch 141/200
 - 1s - loss: 36.1263 - val_loss: 36.2371
Epoch 142/200
 - 1s - loss: 36.1207 - val_loss: 36.2856
Epoch 143/200
 - 1s - loss: 36.1268 - val_loss: 36.0067
Epoch 144/200
 - 2s - loss: 36.0543 - val_loss: 36.1193
Epoch 145/200
 - 2s - loss: 36.0496 - val_loss: 36.1894
Epoch 146/200
 - 1s - loss: 36.0944 - val_loss: 35.9033
Epoch 147/200
 - 1s - loss: 36.0458 - val_loss: 35.9434
Epoch 148/200
 - 2s - loss: 36.0495 - val_loss: 36.0443
Epoch 149/200
 - 2s - loss: 36.0337 - val_loss: 36.0869
Epoch 150/200
 - 2s - loss: 36.0383 - val_loss: 35.9639
Epoch 151/200
 - 2s - loss: 36.1032 - val_loss: 36.4199
Epoch 152/200
 - 2s - loss: 35.9911 - val_loss: 35.7753
Epoch 153/200
 - 1s - loss: 35.9812 - val_loss: 36.0435
Epoch 154/200
 - 1s - loss: 35.9759 - val_loss: 36.5610
Epoch 155/200
 - 2s - loss: 35.9243 - val_loss: 35.7853
Epoch 156/200
 - 2s - loss: 35.8941 - val_loss: 35.6839
Epoch 157/200
 - 2s - loss: 35.9137 - val_loss: 35.9285
Epoch 158/200
 - 2s - loss: 35.9725 - val_loss: 36.0955
Epoch 159/200
 - 1s - loss: 35.9078 - val_loss: 36.1670
Epoch 160/200
 - 2s - loss: 35.9367 - val_loss: 35.8694
Epoch 161/200
 - 2s - loss: 35.9811 - val_loss: 35.8200
Epoch 162/200
 - 2s - loss: 35.9015 - val_loss: 35.7984
Epoch 163/200
 - 2s - loss: 35.9173 - val_loss: 36.0742
Epoch 164/200
 - 2s - loss: 35.9011 - val_loss: 36.0201
Epoch 165/200
 - 2s - loss: 35.9234 - val_loss: 36.0285
Epoch 166/200
 - 2s - loss: 35.8952 - val_loss: 36.1086
Epoch 167/200
 - 2s - loss: 35.9092 - val_loss: 35.6452
Epoch 168/200
 - 2s - loss: 35.9498 - val_loss: 36.0514
Epoch 169/200
 - 1s - loss: 35.8833 - val_loss: 35.6139
Epoch 170/200
 - 2s - loss: 35.9420 - val_loss: 35.8190
Epoch 171/200
 - 2s - loss: 35.8770 - val_loss: 35.8622
Epoch 172/200
 - 1s - loss: 35.8725 - val_loss: 35.7829
Epoch 173/200
 - 2s - loss: 35.9511 - val_loss: 35.9187
Epoch 174/200
 - 2s - loss: 35.9248 - val_loss: 35.8430
Epoch 175/200
 - 1s - loss: 35.8914 - val_loss: 35.6350
Epoch 176/200
 - 2s - loss: 35.9157 - val_loss: 35.8347
Epoch 177/200
 - 1s - loss: 35.9257 - val_loss: 36.9883
Epoch 178/200
 - 1s - loss: 35.9499 - val_loss: 35.8741
Epoch 179/200
 - 2s - loss: 35.8640 - val_loss: 35.7040
Epoch 180/200
 - 2s - loss: 35.9668 - val_loss: 36.3166
Epoch 181/200
 - 1s - loss: 35.9108 - val_loss: 35.7855
Epoch 182/200
 - 2s - loss: 35.9783 - val_loss: 35.9752
Epoch 183/200
 - 2s - loss: 35.9040 - val_loss: 35.9314
Epoch 184/200
 - 2s - loss: 35.9860 - val_loss: 35.8361
Epoch 185/200
 - 2s - loss: 35.9007 - val_loss: 35.9621
Epoch 186/200
 - 2s - loss: 35.9427 - val_loss: 36.1603
Epoch 187/200
 - 2s - loss: 35.9403 - val_loss: 35.8693
Epoch 188/200
 - 2s - loss: 35.8942 - val_loss: 35.8777
Epoch 189/200
 - 2s - loss: 35.9350 - val_loss: 35.7134
Epoch 190/200
 - 2s - loss: 35.9265 - val_loss: 35.7824
Epoch 191/200
 - 2s - loss: 35.9642 - val_loss: 35.9603
Epoch 192/200
 - 2s - loss: 35.9724 - val_loss: 35.8239
Epoch 193/200
 - 2s - loss: 35.9766 - val_loss: 35.8613
Epoch 194/200
 - 1s - loss: 35.9019 - val_loss: 36.1024
Epoch 195/200
 - 2s - loss: 35.8856 - val_loss: 36.0546
Epoch 196/200
 - 2s - loss: 35.9531 - val_loss: 35.9273
Epoch 197/200
 - 1s - loss: 35.9851 - val_loss: 35.7515
Epoch 198/200
 - 1s - loss: 35.9071 - val_loss: 35.7996
Epoch 199/200
 - 2s - loss: 35.9672 - val_loss: 35.7946
Epoch 200/200
 - 2s - loss: 35.9373 - val_loss: 35.6854
>>> model.get_weights()
[array([[ 0.00290514,  0.00262838,  0.0030495 , -0.00168228,  0.00210304,
        -0.00201239,  0.0028883 , -0.00056165],
       [ 0.00234348,  0.00199531,  0.00077818, -0.00050342,  0.0020441 ,
         0.00112457, -0.00049605, -0.00200256],
       [ 0.00270921,  0.00254747, -0.00060978, -0.00288749,  0.00015744,
        -0.00284515, -0.00275534, -0.00313805],
       [ 0.00194611, -0.00282951, -0.00058111, -0.00109371,  0.00135348,
        -0.00315483,  0.00073414,  0.00213229]], dtype=float32), array([[ 1.2923763e-03,  1.1368622e-03,  1.5199231e-03,  2.8875691e-04,
         1.5117979e-03,  1.3037613e-03, -2.9647208e-03, -2.9811985e-03],
       [-1.0149063e-03,  2.9482027e-03, -4.9208300e-03,  9.2176895e-04,
        -1.0394498e-01, -2.6231050e-03,  1.4443670e-03,  2.7571335e-03],
       [-2.8762526e-03,  2.0027633e-03,  2.0314588e-03,  5.0777907e-04,
         1.2382147e-03, -5.8319455e-04, -1.6004947e-03, -1.5543890e-05],
       [-8.4275001e-04, -2.3687249e-03, -3.1350306e-03, -6.2337797e-04,
        -2.2207608e-04,  2.7710159e-04,  1.7321456e-03,  5.7641807e-04],
       [-2.1701148e-03, -7.1024318e-04, -1.1765476e-03,  2.6775540e-03,
        -2.2052119e-03,  2.7831120e-03, -2.7507730e-03, -2.8825167e-03],
       [-3.0429554e-03,  2.2471454e-03, -5.2718790e-03, -3.0438125e-03,
         3.9936596e-04, -3.2651199e-03,  3.8901297e-04,  2.4900199e-03],
       [-3.1501292e-03,  2.6483014e-03, -2.7623808e-03, -2.8582136e-03,
         2.1490082e-03,  3.0294126e-03, -2.7346159e-03, -1.4050120e-03],
       [ 1.8142608e-03,  2.2236782e-05, -5.0621442e-03, -2.4004821e-03,
        -2.3722358e-05, -2.5786243e-03,  1.4607591e-03,  1.0332890e-04]],
      dtype=float32), array([[ 2.6561096e-03, -5.7853886e-04, -3.0732220e-03, -1.3465824e-03,
        -9.2994998e-04, -2.0410786e-03, -2.7646224e-03,  1.2418604e-06],
       [ 7.8738399e-04, -1.2329174e-03, -1.9871953e-03,  8.9196186e-04,
         2.5951935e-03, -2.8609494e-03, -5.9201417e-04, -2.9613063e-04],
       [ 1.1387704e-03,  1.2542006e-03,  2.8023641e-03, -1.5356420e-03,
         2.8979536e-03, -3.1568056e-03, -1.0105469e-03,  1.8577536e-03],
       [ 2.9285031e-03,  3.3749492e-04, -1.8558317e-03, -1.0673651e-03,
         1.1508723e-03,  2.1153335e-03, -2.5689362e-03, -1.7218103e-03],
       [ 2.4363587e-03, -9.1877871e-04,  1.5239965e-03, -1.6719893e-03,
        -1.6097681e-03, -2.4662833e-03,  9.3336625e-04,  2.1729157e-03],
       [ 9.2242204e-04,  2.9324261e-03, -2.0409457e-04, -1.8995573e-03,
         1.7225994e-03,  1.5434787e-03,  2.6109489e-03,  2.3047072e-03],
       [ 6.5941742e-04, -5.1042996e-05, -2.0764029e-04, -1.4987784e-03,
         9.5799449e-05, -2.3403065e-03,  1.0377952e-03,  8.7739038e-04],
       [-4.3365982e-04,  2.9515133e-03, -2.2708126e-03,  4.9283775e-04,
        -2.0128759e-03, -2.5962404e-04, -2.6742637e-03, -1.0409714e-03],
       [ 2.6764420e-03, -3.0156011e-03,  1.4317103e-03,  2.9957676e-03,
        -2.3548743e-03, -2.9054254e-03,  2.2916268e-03,  7.1272120e-04],
       [ 1.2640493e-03, -1.5469814e-03, -2.4988523e-03, -2.7210133e-03,
         2.1379457e-03,  2.7727950e-03,  1.0158336e-03,  2.4865457e-04],
       [-2.8168869e-03,  4.9973336e-01, -1.5401843e-03,  2.4117080e-03,
         2.9510541e-03, -3.0291497e-03,  5.0234830e-01,  2.8097909e-03],
       [ 9.4219344e-04, -1.1045306e-03, -2.0650885e-04,  9.6566777e-04,
         5.6411978e-04,  2.0064574e-03, -9.6075272e-04, -2.7417492e-03],
       [ 3.8881661e-04, -5.0099188e-01, -1.0531358e-03,  2.3715389e-03,
         1.6297039e-05,  3.1132882e-03, -5.0271624e-01,  5.3682900e-04],
       [-1.8612046e-03,  1.5373257e-04, -2.0733792e-03, -3.0785527e-03,
         1.7610210e-03, -1.9779818e-03,  1.2710022e-03, -1.8253775e-03],
       [-2.4196201e-03, -1.6779231e-03, -2.9139267e-03,  9.0201851e-04,
         3.0990285e-03, -1.4463239e-03,  7.8480254e-04,  5.0801260e-05],
       [ 2.0743464e-04, -1.2115710e-03, -2.8438233e-03,  2.6229254e-03,
        -2.8107299e-03,  5.3362467e-04,  3.1246315e-03, -2.2125756e-04]],
      dtype=float32), array([[0.00104829],
       [0.00056285],
       [0.00044203],
       [0.00112549],
       [0.00164167],
       [0.00096661],
       [0.00057826],
       [0.00155327]], dtype=float32)]
>>> y_pred=model.predict(x_test)
>>> complex_mean_squared_error(y_test,y_pred)
<tf.Tensor: id=315931, shape=(), dtype=float64, numpy=0.5133283892695609>
>>> tf.math.log(2)
Traceback (most recent call last):
  File "<pyshell#475>", line 1, in <module>
    tf.math.log(2)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 5866, in log
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InternalError: Could not find valid device for node.
Node: {{node Log}}
All kernels registered for op Log :
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_COMPLEX128]
 [Op:Log]
>>> tf.math.log(tf.Variable(2))
Traceback (most recent call last):
  File "<pyshell#476>", line 1, in <module>
    tf.math.log(tf.Variable(2))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 5866, in log
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InternalError: Could not find valid device for node.
Node: {{node Log}}
All kernels registered for op Log :
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_COMPLEX128]
 [Op:Log]
>>> tf.math.log(tf.Variable([2]))
Traceback (most recent call last):
  File "<pyshell#477>", line 1, in <module>
    tf.math.log(tf.Variable([2]))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 5866, in log
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InternalError: Could not find valid device for node.
Node: {{node Log}}
All kernels registered for op Log :
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_COMPLEX128]
 [Op:Log]
>>> tf.math.log(tf.Variable(np.arry([2])))
Traceback (most recent call last):
  File "<pyshell#478>", line 1, in <module>
    tf.math.log(tf.Variable(np.arry([2])))
AttributeError: module 'numpy' has no attribute 'arry'
>>> tf.math.log(tf.Variable(np.array([2])))
Traceback (most recent call last):
  File "<pyshell#479>", line 1, in <module>
    tf.math.log(tf.Variable(np.array([2])))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 5866, in log
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InternalError: Could not find valid device for node.
Node: {{node Log}}
All kernels registered for op Log :
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_COMPLEX128]
 [Op:Log]
>>> a = tf.Variable(np.array(2))
>>> a
<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=2>
>>> tf.math.log(a)
Traceback (most recent call last):
  File "<pyshell#482>", line 1, in <module>
    tf.math.log(a)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 5866, in log
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InternalError: Could not find valid device for node.
Node: {{node Log}}
All kernels registered for op Log :
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_COMPLEX128]
 [Op:Log]
>>> tf.math.log(tf.Variable(np.array(2.)))
<tf.Tensor: id=315976, shape=(), dtype=float64, numpy=0.6931471805599453>
>>> tf.math.log(tf.Variable(np.array(2.7.)))
SyntaxError: invalid syntax
>>> tf.math.log(tf.Variable(np.array(2.7)))
<tf.Tensor: id=315986, shape=(), dtype=float64, numpy=0.9932517730102834>
>>> tf.math.log(tf.Variable(np.array(10.)))
<tf.Tensor: id=315996, shape=(), dtype=float64, numpy=2.302585092994046>
>>> tf.subtract(a,a)
<tf.Tensor: id=316000, shape=(), dtype=int32, numpy=0>
>>> tf.subtract(a,2*a)
<tf.Tensor: id=316006, shape=(), dtype=int32, numpy=-2>
>>> np.exp(-20)
2.061153622438558e-09
>>> weight_matrix = model.get_weights()[1]
>>> weight_matrix
array([[ 1.2923763e-03,  1.1368622e-03,  1.5199231e-03,  2.8875691e-04,
         1.5117979e-03,  1.3037613e-03, -2.9647208e-03, -2.9811985e-03],
       [-1.0149063e-03,  2.9482027e-03, -4.9208300e-03,  9.2176895e-04,
        -1.0394498e-01, -2.6231050e-03,  1.4443670e-03,  2.7571335e-03],
       [-2.8762526e-03,  2.0027633e-03,  2.0314588e-03,  5.0777907e-04,
         1.2382147e-03, -5.8319455e-04, -1.6004947e-03, -1.5543890e-05],
       [-8.4275001e-04, -2.3687249e-03, -3.1350306e-03, -6.2337797e-04,
        -2.2207608e-04,  2.7710159e-04,  1.7321456e-03,  5.7641807e-04],
       [-2.1701148e-03, -7.1024318e-04, -1.1765476e-03,  2.6775540e-03,
        -2.2052119e-03,  2.7831120e-03, -2.7507730e-03, -2.8825167e-03],
       [-3.0429554e-03,  2.2471454e-03, -5.2718790e-03, -3.0438125e-03,
         3.9936596e-04, -3.2651199e-03,  3.8901297e-04,  2.4900199e-03],
       [-3.1501292e-03,  2.6483014e-03, -2.7623808e-03, -2.8582136e-03,
         2.1490082e-03,  3.0294126e-03, -2.7346159e-03, -1.4050120e-03],
       [ 1.8142608e-03,  2.2236782e-05, -5.0621442e-03, -2.4004821e-03,
        -2.3722358e-05, -2.5786243e-03,  1.4607591e-03,  1.0332890e-04]],
      dtype=float32)
>>> tf.clip(weight_matrix,np.exp(-10),inf)
Traceback (most recent call last):
  File "<pyshell#492>", line 1, in <module>
    tf.clip(weight_matrix,np.exp(-10),inf)
AttributeError: module 'tensorflow' has no attribute 'clip'
>>> tf.clip_by_value(weight_matrix,np.exp(-10),inf)
Traceback (most recent call last):
  File "<pyshell#493>", line 1, in <module>
    tf.clip_by_value(weight_matrix,np.exp(-10),inf)
NameError: name 'inf' is not defined
>>> tf.clip_by_value(weight_matrix,np.exp(-10),float('inf'))
<tf.Tensor: id=316020, shape=(8, 8), dtype=float32, numpy=
array([[1.2923763e-03, 1.1368622e-03, 1.5199231e-03, 2.8875691e-04,
        1.5117979e-03, 1.3037613e-03, 4.5399931e-05, 4.5399931e-05],
       [4.5399931e-05, 2.9482027e-03, 4.5399931e-05, 9.2176895e-04,
        4.5399931e-05, 4.5399931e-05, 1.4443670e-03, 2.7571335e-03],
       [4.5399931e-05, 2.0027633e-03, 2.0314588e-03, 5.0777907e-04,
        1.2382147e-03, 4.5399931e-05, 4.5399931e-05, 4.5399931e-05],
       [4.5399931e-05, 4.5399931e-05, 4.5399931e-05, 4.5399931e-05,
        4.5399931e-05, 2.7710159e-04, 1.7321456e-03, 5.7641807e-04],
       [4.5399931e-05, 4.5399931e-05, 4.5399931e-05, 2.6775540e-03,
        4.5399931e-05, 2.7831120e-03, 4.5399931e-05, 4.5399931e-05],
       [4.5399931e-05, 2.2471454e-03, 4.5399931e-05, 4.5399931e-05,
        3.9936596e-04, 4.5399931e-05, 3.8901297e-04, 2.4900199e-03],
       [4.5399931e-05, 2.6483014e-03, 4.5399931e-05, 4.5399931e-05,
        2.1490082e-03, 3.0294126e-03, 4.5399931e-05, 4.5399931e-05],
       [1.8142608e-03, 4.5399931e-05, 4.5399931e-05, 4.5399931e-05,
        4.5399931e-05, 4.5399931e-05, 1.4607591e-03, 1.0332890e-04]],
      dtype=float32)>
>>> np.exp(-10)
4.5399929762484854e-05
>>> tf.where(weight_matrix>np.exp(-10),weight_matrix,tf.zeros_like(weight_matrix))
<tf.Tensor: id=316026, shape=(8, 8), dtype=float32, numpy=
array([[0.00129238, 0.00113686, 0.00151992, 0.00028876, 0.0015118 ,
        0.00130376, 0.        , 0.        ],
       [0.        , 0.0029482 , 0.        , 0.00092177, 0.        ,
        0.        , 0.00144437, 0.00275713],
       [0.        , 0.00200276, 0.00203146, 0.00050778, 0.00123821,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.0002771 , 0.00173215, 0.00057642],
       [0.        , 0.        , 0.        , 0.00267755, 0.        ,
        0.00278311, 0.        , 0.        ],
       [0.        , 0.00224715, 0.        , 0.        , 0.00039937,
        0.        , 0.00038901, 0.00249002],
       [0.        , 0.0026483 , 0.        , 0.        , 0.00214901,
        0.00302941, 0.        , 0.        ],
       [0.00181426, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00146076, 0.00010333]], dtype=float32)>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 132, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1', kernel_regularizer=llog)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 67, in call
    outputs = tf.matmul(inputs,tf.cast(tf.complex(reLogInv(self.kernel),tf.zeros_like(self.kernel)),tf.complex128))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 482, in complex
    real = ops.convert_to_tensor(real, name="real")
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1100, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 284, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 455, in make_tensor_proto
    raise ValueError("None values not supported.")
ValueError: None values not supported.
>>> p_dense_1.kernel
Traceback (most recent call last):
  File "<pyshell#497>", line 1, in <module>
    p_dense_1.kernel
NameError: name 'p_dense_1' is not defined
>>> value_in.kernel
Traceback (most recent call last):
  File "<pyshell#498>", line 1, in <module>
    value_in.kernel
AttributeError: 'Tensor' object has no attribute 'kernel'
>>> model.get_weights()
Traceback (most recent call last):
  File "<pyshell#499>", line 1, in <module>
    model.get_weights()
NameError: name 'model' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 136, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1', kernel_regularizer=llog)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 69, in call
    a = tf.complex(reLogInv(self.kernel),tf.zeros_like(self.kernel))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 482, in complex
    real = ops.convert_to_tensor(real, name="real")
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1100, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 284, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 455, in make_tensor_proto
    raise ValueError("None values not supported.")
ValueError: None values not supported.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
20
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 135, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1', kernel_regularizer=llog)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 68, in call
    a = tf.complex(reLogInv(self.kernel),tf.zeros_like(self.kernel))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 482, in complex
    real = ops.convert_to_tensor(real, name="real")
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1100, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 284, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 455, in make_tensor_proto
    raise ValueError("None values not supported.")
ValueError: None values not supported.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
<tf.Variable 'power_dense_1/kernel:0' shape=(4, 8) dtype=float32>
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 135, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1', kernel_regularizer=llog)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 68, in call
    a = tf.complex(reLogInv(self.kernel),tf.zeros_like(self.kernel))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 482, in complex
    real = ops.convert_to_tensor(real, name="real")
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1100, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 284, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 455, in make_tensor_proto
    raise ValueError("None values not supported.")
ValueError: None values not supported.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 134, in <module>
    p_dense_1 = PowerDense(8, True, name='power_dense_1', kernel_regularizer=llog)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 489, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 68, in call
    a = tf.complex(reLogInv(self.kernel),tf.zeros_like(self.kernel))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 482, in complex
    real = ops.convert_to_tensor(real, name="real")
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1100, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 284, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 455, in make_tensor_proto
    raise ValueError("None values not supported.")
ValueError: None values not supported.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> kernel = model.getWeights()[1]
Traceback (most recent call last):
  File "<pyshell#500>", line 1, in <module>
    kernel = model.getWeights()[1]
AttributeError: 'Model' object has no attribute 'getWeights'
>>> kernel = model.get_weights()[1]
>>> kernel
array([[ 0.3591349 , -0.0934878 , -0.14452913, -0.08963162, -0.33718297,
        -0.2931434 ,  0.54980224,  0.21311134],
       [ 0.31063706,  0.05836475,  0.24178272, -0.5514083 , -0.13993856,
        -0.4951055 , -0.5908557 , -0.2644022 ],
       [ 0.34372628, -0.22257105, -0.08339763,  0.11733603, -0.3248904 ,
         0.07091093,  0.6024218 ,  0.58936256],
       [ 0.02256548, -0.2804645 ,  0.38157094,  0.40101844,  0.39291865,
        -0.09750283,  0.53166956, -0.38764733],
       [-0.44331443, -0.19677907, -0.39012423, -0.58885974,  0.598401  ,
         0.3599273 , -0.4719804 , -0.44372863],
       [ 0.25791335,  0.54410857,  0.24060917, -0.22007078,  0.34887183,
         0.04940486,  0.01525724,  0.5917091 ],
       [-0.49465948,  0.24444562,  0.17890507, -0.27059352, -0.00689137,
         0.45860714, -0.02425385,  0.596196  ],
       [-0.2105569 , -0.03595817, -0.29085526,  0.49332803,  0.30567992,
        -0.33681476, -0.13618854, -0.598612  ]], dtype=float32)
>>> reLogInv(kernel)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 6, in <module>
    from . import conv_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\conv_utils.py", line 9, in <module>
    from .. import backend as K
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\__init__.py", line 1, in <module>
    from .load_backend import epsilon
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\load_backend.py", line 90, in <module>
    from .tensorflow_backend import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 5, in <module>
    import tensorflow as tf
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\__init__.py", line 40, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\__init__.py", line 83, in <module>
    from tensorflow.python import keras
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\__init__.py", line 27, in <module>
    from tensorflow.python.keras import applications
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\applications\__init__.py", line 22, in <module>
    import keras_applications
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras_applications\__init__.py", line 60, in <module>
    from . import mobilenet_v2
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
  File "<frozen importlib._bootstrap_external>", line 764, in get_code
  File "<frozen importlib._bootstrap_external>", line 832, in get_data
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 151, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_squared_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_2478]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Tensor("metrics/complex_mean_absolute_error/Mean:0", shape=(), dtype=float32)
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 152, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_absolute_error/Mean_1 (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_2482]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Tensor("loss/dense_2_loss/complex_absolute_error/Mean:0", shape=(), dtype=float64)
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 152, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_absolute_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_2482]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Tensor("loss/dense_2_loss/complex_absolute_error/Mean:0", shape=(), dtype=float64)
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
 - 2s - loss: nan - val_loss: nan
Epoch 4/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 152, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> y_pred = model.predict(x_train)
>>> np.isnan(x_train).any()
False
>>> np.isnan(x_train).all()
False
>>> y_pred.shape
(8000, 1, 2)
>>> y_pred.shape[:10]
(8000, 1, 2)
>>> y_pred[:10]
array([[[ 0.75107251, -1.23032069]],

       [[ 0.16613643, -1.08965419]],

       [[-1.03869403,  0.41293365]],

       [[ 0.00647964, -0.99355   ]],

       [[ 0.58239367, -1.11629016]],

       [[ 0.4170628 , -0.59918683]],

       [[ 0.5107355 , -0.54183273]],

       [[-0.61471227, -0.43638726]],

       [[-1.03518211,  0.41581387]],

       [[-0.98353812,  0.26893257]]])
>>> loss=complex_absolute_error(y_train, y_pred)
>>> loss.shape
TensorShape([8000, 1, 2])
>>> loss[:10]
<tf.Tensor: id=3642, shape=(10, 1, 2), dtype=float64, numpy=
array([[[0.65205404, 1.23032069]],

       [[0.03277573, 1.08965419]],

       [[1.5796639 , 0.41293365]],

       [[0.28939849, 0.99355   ]],

       [[0.4035381 , 1.11629016]],

       [[1.00702704, 0.59918683]],

       [[0.99557302, 0.54183273]],

       [[2.45201599, 0.43638726]],

       [[1.62343408, 0.41581387]],

       [[1.65131112, 0.26893257]]])>
>>> np.isnan(loss).any()
False
>>> np.isnan(loss).all()
False
>>> np.isnan([True,False]).all()
False
>>> np.isnan([True,False]).any()
False
>>> np.isnan([1/0,1]).any()
Traceback (most recent call last):
  File "<pyshell#517>", line 1, in <module>
    np.isnan([1/0,1]).any()
ZeroDivisionError: division by zero
>>> np.isnan([float('nana'),1]).any()
Traceback (most recent call last):
  File "<pyshell#518>", line 1, in <module>
    np.isnan([float('nana'),1]).any()
ValueError: could not convert string to float: 'nana'
>>> np.isnan([float('nan'),1]).any()
True
>>> np.sum(loss)
4.8856096318807e+43
>>> exp(70)
Traceback (most recent call last):
  File "<pyshell#521>", line 1, in <module>
    exp(70)
NameError: name 'exp' is not defined
>>> np.exp(70)
2.515438670919167e+30
>>> max(loss)
Traceback (most recent call last):
  File "<pyshell#523>", line 1, in <module>
    max(loss)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 972, in __bool__
    return bool(self.numpy())
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
>>> np.max(loss)
1.221402407970175e+43
>>> 4*np.max(loss)
4.8856096318807e+43
>>> np.sum(loss)-4*np.max(loss)
0.0
>>> np.argmax(loss)
4586
>>> loss[4586]
<tf.Tensor: id=3657, shape=(1, 2), dtype=float64, numpy=array([[0.3292804 , 0.13623473]])>
>>> loss[4586/2]
Traceback (most recent call last):
  File "<pyshell#529>", line 1, in <module>
    loss[4586/2]
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\array_ops.py", line 644, in _slice_helper
    _check_index(s)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\array_ops.py", line 535, in _check_index
    raise TypeError(_SLICE_TYPE_ERROR + ", got {!r}".format(idx))
TypeError: Only integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid indices, got 2293.0
>>> loss[int(4586/2)]
<tf.Tensor: id=3662, shape=(1, 2), dtype=float64, numpy=array([[1.22140241e+43, 1.22140241e+43]])>
>>> 4586/2
2293.0
>>> loss[2293]
<tf.Tensor: id=3667, shape=(1, 2), dtype=float64, numpy=array([[1.22140241e+43, 1.22140241e+43]])>
>>> x_train[2293]
array([-9.11285643e-06, -7.89277350e-01])
>>> model.predict(x_train[2293:2294])
array([[[-1.22140241e+43, -1.22140241e+43]]])
>>> np.exp(100)
2.6881171418161356e+43
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "<pyshell#536>", line 1, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 192, in fit_loop
    callbacks._call_batch_hook('train', 'begin', batch_index, batch_logs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\callbacks\callbacks.py", line 88, in _call_batch_hook
    delta_t_median = np.median(self._delta_ts[hook_name])
  File "<__array_function__ internals>", line 6, in median
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3502, in median
    overwrite_input=overwrite_input)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3410, in _ureduce
    r = func(a, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\function_base.py", line 3535, in _median
    part = partition(a, kth, axis=axis)
  File "<__array_function__ internals>", line 6, in partition
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\fromnumeric.py", line 745, in partition
    a.partition(kth, axis=axis, kind=kind, order=order)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 136, in <module>
    p_dense_1 = PowerDense(8, True, copy_log = False, name='power_dense_1', kernel_regularizer=llog)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 463, in __call__
    self.build(unpack_singleton(input_shapes))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 83, in build
    if self.use_log and self.copy_log:
AttributeError: 'PowerDense' object has no attribute 'copy_log'
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              16        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              64        
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 152
Trainable params: 152
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: 28.1807 - val_loss: 27.1888
Epoch 2/50
 - 2s - loss: 26.4575 - val_loss: 25.6262
Epoch 3/50
 - 2s - loss: 25.1078 - val_loss: 24.4175
Epoch 4/50
 - 1s - loss: 23.7713 - val_loss: 23.3952
Epoch 5/50
 - 1s - loss: 22.7569 - val_loss: 22.1065
Epoch 6/50
 - 2s - loss: 21.6909 - val_loss: 21.3690
Epoch 7/50
 - 2s - loss: 21.0513 - val_loss: 20.7603
Epoch 8/50
 - 2s - loss: 20.7794 - val_loss: 20.7814
Epoch 9/50
 - 2s - loss: 20.6870 - val_loss: 20.4056
Epoch 10/50
 - 2s - loss: 20.6842 - val_loss: 20.6201
Epoch 11/50
 - 2s - loss: 20.6801 - val_loss: 20.7436
Epoch 12/50
 - 2s - loss: 20.6411 - val_loss: 20.7690
Epoch 13/50
 - 2s - loss: 20.6766 - val_loss: 20.7295
Epoch 14/50
 - 2s - loss: 20.6706 - val_loss: 20.4268
Epoch 15/50
 - 2s - loss: 20.6428 - val_loss: 20.6426
Epoch 16/50
 - 2s - loss: 20.6448 - val_loss: 20.4491
Epoch 17/50
 - 2s - loss: 20.6517 - val_loss: 20.6258
Epoch 18/50
 - 2s - loss: 20.6721 - val_loss: 20.7966
Epoch 19/50
 - 2s - loss: 20.6594 - val_loss: 20.6121
Epoch 20/50
 - 2s - loss: 20.6706 - val_loss: 20.5697
Epoch 21/50
 - 2s - loss: 20.6544 - val_loss: 20.5432
Epoch 22/50
 - 2s - loss: 20.6448 - val_loss: 20.5143
Epoch 23/50
 - 2s - loss: 20.6734 - val_loss: 20.6763
Epoch 24/50
 - 2s - loss: 20.6656 - val_loss: 20.7746
Epoch 25/50
 - 2s - loss: 20.6419 - val_loss: 20.6238
Epoch 26/50
 - 2s - loss: 20.6454 - val_loss: 20.6293
Epoch 27/50
 - 2s - loss: 20.7050 - val_loss: 20.6877
Epoch 28/50
 - 2s - loss: 20.6943 - val_loss: 20.7263
Epoch 29/50
 - 2s - loss: 20.6504 - val_loss: 20.8494
Epoch 30/50
 - 2s - loss: 20.6654 - val_loss: 20.7023
Epoch 31/50
 - 2s - loss: 20.6713 - val_loss: 20.7479
Epoch 32/50
 - 2s - loss: 20.6377 - val_loss: 20.7694
Epoch 33/50
 - 2s - loss: 20.6899 - val_loss: 20.7426
Epoch 34/50
 - 2s - loss: 20.6671 - val_loss: 20.8183
Epoch 35/50
 - 2s - loss: 20.6812 - val_loss: 20.7275
Epoch 36/50
 - 2s - loss: 20.6559 - val_loss: 20.8297
Epoch 37/50
 - 2s - loss: 20.6630 - val_loss: 20.7589
Epoch 38/50
 - 2s - loss: 20.6930 - val_loss: 20.6651
Epoch 39/50
 - 2s - loss: 20.6959 - val_loss: 20.5906
Epoch 40/50
 - 2s - loss: 20.6573 - val_loss: 20.5182
Epoch 41/50
 - 2s - loss: 20.6222 - val_loss: 20.4697
Epoch 42/50
 - 2s - loss: 20.6554 - val_loss: 20.6157
Epoch 43/50
 - 2s - loss: 20.6815 - val_loss: 20.4921
Epoch 44/50
 - 2s - loss: 20.6604 - val_loss: 20.5368
Epoch 45/50
 - 2s - loss: 20.6511 - val_loss: 20.7125
Epoch 46/50
 - 2s - loss: 20.6607 - val_loss: 20.6243
Epoch 47/50
 - 2s - loss: 20.6749 - val_loss: 20.5268
Epoch 48/50
 - 2s - loss: 20.6990 - val_loss: 20.7888
Epoch 49/50
 - 2s - loss: 20.6550 - val_loss: 20.6936
Epoch 50/50
 - 2s - loss: 20.6473 - val_loss: 20.6687
<keras.callbacks.callbacks.History object at 0x0000025114D979E8>
>>> y_pred = model.predict(x_test)
>>> complex_mean_absolute_error(y_train,t_pred)
Traceback (most recent call last):
  File "<pyshell#539>", line 1, in <module>
    complex_mean_absolute_error(y_train,t_pred)
NameError: name 't_pred' is not defined
>>> complex_mean_absolute_error(y_train,y_pred)
Traceback (most recent call last):
  File "<pyshell#540>", line 1, in <module>
    complex_mean_absolute_error(y_train,y_pred)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 132, in complex_mean_absolute_error
    return K.mean(tf.abs(y_pred - y_true))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 884, in binary_op_wrapper
    return func(x, y, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 11571, in sub
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [2000,1,2] vs. [8000,1,2] [Op:Sub] name: sub/
>>> y_pred = tf.convert_to_tensor(y_pred)
>>> y_true = tf.stack((y_true,tf.zeros_like(y_true)),axis=2)
Traceback (most recent call last):
  File "<pyshell#542>", line 1, in <module>
    y_true = tf.stack((y_true,tf.zeros_like(y_true)),axis=2)
NameError: name 'y_true' is not defined
>>> complex_mean_absolute_error(y_test,y_pred)
<tf.Tensor: id=81310, shape=(), dtype=float64, numpy=0.275130863345746>
>>> y_pred[:10]
<tf.Tensor: id=81315, shape=(10, 1, 2), dtype=float64, numpy=
array([[[0.5364459 , 0.01178476]],

       [[0.53660631, 0.00574349]],

       [[0.53659953, 0.00574342]],

       [[0.53660506, 0.00444888]],

       [[0.53645293, 0.01178519]],

       [[0.53644776, 0.01178484]],

       [[0.53655359, 0.00574292]],

       [[0.53657854, 0.00444998]],

       [[0.53658191, 0.00444934]],

       [[0.53656913, 0.00444021]]])>
>>> x_test[:10]
array([[-0.44093031,  0.92464397],
       [ 0.17546333,  0.0947444 ],
       [ 0.92657208,  0.14290517],
       [-0.54044055, -0.1658563 ],
       [-0.10257923,  0.6188521 ],
       [-0.37739645,  0.85137227],
       [ 0.65314785,  0.69965887],
       [-0.02092837, -0.30060981],
       [-0.23323159, -0.34473413],
       [ 0.63253578, -0.58784354]])
>>> model.get_weights()
[array([[ 0.00039414,  0.00163205,  0.00130415,  0.00207207, -0.00255274,
        -0.00305499, -0.00225479, -0.00206689],
       [-0.00143165,  0.00229958, -0.00110382,  0.00231957, -0.00270134,
         0.00221572,  0.00075945,  0.00046211]], dtype=float32), array([[-7.4737688e-04,  1.5100173e-04,  2.0642665e-03,  1.4627629e-03,
         2.5641927e-03, -3.0016373e-03, -2.5482655e-03,  4.0316867e-04],
       [-2.1211237e-03,  3.1217462e-03,  2.4911168e-03, -1.7666224e-03,
        -2.0527972e-03,  2.1287999e-03,  2.7533900e-03, -2.1084175e-03],
       [-2.9861298e-03, -3.1197355e-03, -1.2508556e-03, -2.6803480e-03,
         1.0303718e-03, -1.4220258e-03, -9.1389613e-04, -4.1877202e-04],
       [-7.9477893e-04, -2.8134108e-04, -2.1095707e-03,  1.0536299e-03,
        -2.0419336e-03,  1.7004116e-03, -1.7323478e-03,  2.5921555e-03],
       [ 2.7764903e-03, -2.4529297e-03, -2.0761103e-03,  2.5616097e-03,
        -2.9925327e-04, -2.3108197e-03, -3.0422972e-03, -1.4776713e-03],
       [ 3.1071983e-04,  3.0088579e-04,  8.1074244e-04,  1.1123241e-03,
        -1.5645631e-03, -2.9224970e-03,  2.7274566e-03,  2.5167658e-03],
       [ 3.6981833e-04,  2.4967149e-03, -2.6369060e-04,  3.1023552e-03,
         1.1470916e-03,  5.3753168e-04, -3.5360281e-05,  4.7970549e-04],
       [-9.7706984e-04, -2.3299600e-03,  2.1005368e-03,  3.7877343e-04,
         6.1598839e-05, -2.9737563e-03,  2.4926226e-04, -5.6071061e-04]],
      dtype=float32), array([[-1.5010620e-03, -2.2457265e-03,  7.4131647e-04,  3.1433876e-03,
         2.3318799e-03, -3.1166943e-03,  6.7062856e-04, -2.8815821e-03],
       [ 2.6354701e-03, -2.8291554e-03,  2.8208243e-03, -2.1286246e-03,
         3.0479496e-03,  1.0697769e-03,  3.0183082e-03, -2.9059851e-03],
       [ 8.4854505e-04, -2.1002851e-03, -3.0666655e-03,  3.0744639e-03,
         6.7896349e-04, -2.9581771e-03, -1.7434329e-03,  1.7114076e-03],
       [ 1.8264283e-05,  3.0903735e-03,  2.0258613e-03,  1.5360558e-03,
        -2.8295198e-03, -3.0323733e-03,  7.4015593e-04, -2.9543340e-03],
       [-6.0667389e-04,  9.0196659e-04,  2.7824491e-03, -8.2925137e-04,
        -1.9757045e-03,  1.7496274e-03, -1.7941704e-03,  2.4290336e-03],
       [ 2.3582357e-03, -6.0009095e-04,  1.3804026e-03, -1.7652519e-03,
         2.1559659e-03, -1.7300644e-03, -2.2312873e-03,  1.8728685e-03],
       [ 2.7360565e-03,  1.2869069e-03,  1.6417963e-03,  6.1553926e-04,
        -2.1208955e-03,  3.0414788e-03,  2.3868117e-03,  1.7972654e-03],
       [ 3.1566876e-03,  4.0818387e-04,  1.0985865e-03, -2.5768206e-04,
        -3.0332117e-03, -2.3680762e-03, -9.7585144e-06,  1.2937501e-03]],
      dtype=float32), array([[-5.6947349e-05],
       [ 1.6353115e-03],
       [-1.5968654e-03],
       [ 2.0565451e-03],
       [ 5.3142512e-01],
       [-1.1102608e-03],
       [-1.1662130e-03],
       [-2.4665212e-03]], dtype=float32)]
>>> [reLogInv(weight) for weight in model.get_weights()]
[<tf.Tensor: id=81340, shape=(2, 8), dtype=float32, numpy=
array([[ 0.00039414,  0.00163205,  0.00130415,  0.00207207, -0.00255274,
        -0.00305499, -0.00225479, -0.00206689],
       [-0.00143165,  0.00229958, -0.00110382,  0.00231957, -0.00270134,
         0.00221572,  0.00075945,  0.00046211]], dtype=float32)>, <tf.Tensor: id=81348, shape=(8, 8), dtype=float32, numpy=
array([[-7.4737688e-04,  1.5100173e-04,  2.0642665e-03,  1.4627629e-03,
         2.5641927e-03, -3.0016373e-03, -2.5482655e-03,  4.0316867e-04],
       [-2.1211237e-03,  3.1217462e-03,  2.4911168e-03, -1.7666224e-03,
        -2.0527972e-03,  2.1287999e-03,  2.7533900e-03, -2.1084175e-03],
       [-2.9861298e-03, -3.1197355e-03, -1.2508556e-03, -2.6803480e-03,
         1.0303718e-03, -1.4220258e-03, -9.1389613e-04, -4.1877202e-04],
       [-7.9477893e-04, -2.8134108e-04, -2.1095707e-03,  1.0536299e-03,
        -2.0419336e-03,  1.7004116e-03, -1.7323478e-03,  2.5921555e-03],
       [ 2.7764903e-03, -2.4529297e-03, -2.0761103e-03,  2.5616097e-03,
        -2.9925327e-04, -2.3108197e-03, -3.0422972e-03, -1.4776713e-03],
       [ 3.1071983e-04,  3.0088579e-04,  8.1074244e-04,  1.1123241e-03,
        -1.5645631e-03, -2.9224970e-03,  2.7274566e-03,  2.5167658e-03],
       [ 3.6981833e-04,  2.4967149e-03, -2.6369060e-04,  3.1023552e-03,
         1.1470916e-03,  5.3753168e-04, -3.5360281e-05,  4.7970549e-04],
       [-9.7706984e-04, -2.3299600e-03,  2.1005368e-03,  3.7877343e-04,
         6.1598839e-05, -2.9737563e-03,  2.4926226e-04, -5.6071061e-04]],
      dtype=float32)>, <tf.Tensor: id=81356, shape=(8, 8), dtype=float32, numpy=
array([[-1.5010620e-03, -2.2457265e-03,  7.4131647e-04,  3.1433876e-03,
         2.3318799e-03, -3.1166943e-03,  6.7062856e-04, -2.8815821e-03],
       [ 2.6354701e-03, -2.8291554e-03,  2.8208243e-03, -2.1286246e-03,
         3.0479496e-03,  1.0697769e-03,  3.0183082e-03, -2.9059851e-03],
       [ 8.4854505e-04, -2.1002851e-03, -3.0666655e-03,  3.0744639e-03,
         6.7896349e-04, -2.9581771e-03, -1.7434329e-03,  1.7114076e-03],
       [ 1.8264283e-05,  3.0903735e-03,  2.0258613e-03,  1.5360558e-03,
        -2.8295198e-03, -3.0323733e-03,  7.4015593e-04, -2.9543340e-03],
       [-6.0667389e-04,  9.0196659e-04,  2.7824491e-03, -8.2925137e-04,
        -1.9757045e-03,  1.7496274e-03, -1.7941704e-03,  2.4290336e-03],
       [ 2.3582357e-03, -6.0009095e-04,  1.3804026e-03, -1.7652519e-03,
         2.1559659e-03, -1.7300644e-03, -2.2312873e-03,  1.8728685e-03],
       [ 2.7360565e-03,  1.2869069e-03,  1.6417963e-03,  6.1553926e-04,
        -2.1208955e-03,  3.0414788e-03,  2.3868117e-03,  1.7972654e-03],
       [ 3.1566876e-03,  4.0818387e-04,  1.0985865e-03, -2.5768206e-04,
        -3.0332117e-03, -2.3680762e-03, -9.7585144e-06,  1.2937501e-03]],
      dtype=float32)>, <tf.Tensor: id=81364, shape=(8, 1), dtype=float32, numpy=
array([[-5.6947349e-05],
       [ 1.6353115e-03],
       [-1.5968654e-03],
       [ 2.0565451e-03],
       [ 5.3142512e-01],
       [-1.1102608e-03],
       [-1.1662130e-03],
       [-2.4665212e-03]], dtype=float32)>]
>>> np.cast(np.exp(40),np.float16)
Traceback (most recent call last):
  File "<pyshell#548>", line 1, in <module>
    np.cast(np.exp(40),np.float16)
TypeError: '_typedict' object is not callable
>>> np.float16
<class 'numpy.float16'>
>>> np.array(exp(40)).astype(np.float16)
Traceback (most recent call last):
  File "<pyshell#550>", line 1, in <module>
    np.array(exp(40)).astype(np.float16)
NameError: name 'exp' is not defined
>>> np.array(np.exp(40)).astype(np.float16)
array(inf, dtype=float16)
>>> np.array(np.exp(40)).astype(np.float32)
array(2.3538527e+17, dtype=float32)
>>> np.log(10**38)
AttributeError: 'int' object has no attribute 'log'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<pyshell#553>", line 1, in <module>
    np.log(10**38)
TypeError: loop of ufunc does not support argument 0 of type int which has no callable log method
>>> np
<module 'numpy' from 'C:\\Users\\joshm\\AppData\\Local\\Programs\\Python\\Python36\\lib\\site-packages\\numpy\\__init__.py'>
>>> np.log(10.**38)
87.49823353377374
>>> np.log(10.**38/20000)
77.59474598123761
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 6, in <module>
    from . import conv_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\conv_utils.py", line 9, in <module>
    from .. import backend as K
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\__init__.py", line 1, in <module>
    from .load_backend import epsilon
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\load_backend.py", line 90, in <module>
    from .tensorflow_backend import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 5, in <module>
    import tensorflow as tf
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\__init__.py", line 45, in <module>
    from tensorflow._api.v2 import compat
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\_api\v2\compat\__init__.py", line 22, in <module>
    from tensorflow._api.v2.compat import v2
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\_api\v2\compat\v2\__init__.py", line 37, in <module>
    from tensorflow._api.v2.compat.v2 import dtypes
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
  File "<frozen importlib._bootstrap_external>", line 764, in get_code
  File "<frozen importlib._bootstrap_external>", line 832, in get_data
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 155, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 6, in <module>
    from . import conv_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\conv_utils.py", line 9, in <module>
    from .. import backend as K
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\__init__.py", line 1, in <module>
    from .load_backend import epsilon
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\load_backend.py", line 90, in <module>
    from .tensorflow_backend import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 5, in <module>
    import tensorflow as tf
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\__init__.py", line 40, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\__init__.py", line 73, in <module>
    from tensorflow.python.ops.standard_ops import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\standard_ops.py", line 104, in <module>
    from tensorflow.python.ops.template import *
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\template.py", line 29, in <module>
    from tensorflow.python.training.tracking import util as trackable_util
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\training\tracking\util.py", line 51, in <module>
    from tensorflow.python.training.tracking import graph_view as graph_view_lib
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\training\tracking\graph_view.py", line 27, in <module>
    from tensorflow.python.training import optimizer as optimizer_v1
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\training\optimizer.py", line 27, in <module>
    from tensorflow.python.distribute import distribute_lib
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\distribute\__init__.py", line 28, in <module>
    from tensorflow.python.distribute.experimental import collective_all_reduce_strategy
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\distribute\experimental\__init__.py", line 25, in <module>
    from tensorflow.python.distribute import tpu_strategy
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\distribute\tpu_strategy.py", line 52, in <module>
    from tensorflow.python.tpu import training_loop
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 951, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 894, in _find_spec
  File "<frozen importlib._bootstrap_external>", line 1157, in find_spec
  File "<frozen importlib._bootstrap_external>", line 1129, in _get_spec
  File "<frozen importlib._bootstrap_external>", line 1241, in find_spec
  File "<frozen importlib._bootstrap_external>", line 82, in _path_stat
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> y_pred=model.predict(x_train)
>>> loss = complex_absolute_error(y_train,y_pred)
>>> sum(loss)
<tf.Tensor: id=43639, shape=(1, 2), dtype=float64, numpy=array([[57078.31989445, 50985.98340156]])>
>>> tf.reduce_sum(loss)
<tf.Tensor: id=43642, shape=(), dtype=float64, numpy=108064.30329600217>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 2, in <module>
    from . import np_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\np_utils.py", line 6, in <module>
    import numpy as np
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\__init__.py", line 140, in <module>
    from . import _distributor_init
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\_distributor_init.py", line 9, in <module>
    from ctypes import WinDLL
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\ctypes\__init__.py", line 523, in <module>
    from ctypes._endian import BigEndianStructure, LittleEndianStructure
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
  File "<frozen importlib._bootstrap_external>", line 764, in get_code
  File "<frozen importlib._bootstrap_external>", line 832, in get_data
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: 41.9547 - val_loss: 39.3441
Epoch 2/50
 - 3s - loss: 37.9361 - val_loss: 36.4005
Epoch 3/50
 - 2s - loss: 35.3689 - val_loss: 34.4924
Epoch 4/50
 - 2s - loss: 33.8518 - val_loss: 33.4803
Epoch 5/50
 - 2s - loss: 33.0803 - val_loss: 32.7747
Epoch 6/50
 - 2s - loss: 32.5476 - val_loss: 32.4025
Epoch 7/50
 - 2s - loss: 32.0305 - val_loss: 31.9438
Epoch 8/50
 - 2s - loss: 31.6532 - val_loss: 31.3559
Epoch 9/50
 - 1s - loss: 31.3997 - val_loss: 31.5193
Epoch 10/50
 - 1s - loss: 31.3747 - val_loss: 31.5167
Epoch 11/50
 - 2s - loss: 31.3901 - val_loss: 31.6129
Epoch 12/50
 - 1s - loss: 31.3780 - val_loss: 31.5251
Epoch 13/50
 - 1s - loss: 31.3578 - val_loss: 31.3704
Epoch 14/50
 - 1s - loss: 31.3633 - val_loss: 31.3105
Epoch 15/50
 - 1s - loss: 31.4069 - val_loss: 31.4874
Epoch 16/50
 - 1s - loss: 31.4270 - val_loss: 31.5326
Epoch 17/50
 - 3s - loss: 31.4011 - val_loss: 31.3059
Epoch 18/50
 - 2s - loss: 31.3667 - val_loss: 31.5913
Epoch 19/50
 - 2s - loss: 31.3921 - val_loss: 31.3516
Epoch 20/50
 - 2s - loss: 31.3901 - val_loss: 31.3925
Epoch 21/50
 - 2s - loss: 31.3556 - val_loss: 31.3628
Epoch 22/50
 - 2s - loss: 31.4044 - val_loss: 31.6346
Epoch 23/50
 - 2s - loss: 31.3975 - val_loss: 31.6202
Epoch 24/50
 - 2s - loss: 31.3458 - val_loss: 31.3206
Epoch 25/50
 - 3s - loss: 31.3946 - val_loss: 31.3071
Epoch 26/50
 - 2s - loss: 31.4399 - val_loss: 31.4614
Epoch 27/50
 - 2s - loss: 31.4043 - val_loss: 31.2719
Epoch 28/50
 - 1s - loss: 31.4007 - val_loss: 31.4996
Epoch 29/50
 - 1s - loss: 31.3897 - val_loss: 31.2254
Epoch 30/50
 - 2s - loss: 31.3929 - val_loss: 31.2224
Epoch 31/50
 - 2s - loss: 31.3698 - val_loss: 31.4347
Epoch 32/50
 - 2s - loss: 31.4260 - val_loss: 31.3960
Epoch 33/50
 - 2s - loss: 31.4018 - val_loss: 31.5240
Epoch 34/50
 - 2s - loss: 31.4103 - val_loss: 31.2282
Epoch 35/50
 - 2s - loss: 31.3467 - val_loss: 31.5107
Epoch 36/50
 - 2s - loss: 31.4136 - val_loss: 31.4531
Epoch 37/50
 - 2s - loss: 31.4231 - val_loss: 31.1642
Epoch 38/50
 - 3s - loss: 31.4001 - val_loss: 31.5245
Epoch 39/50
 - 2s - loss: 31.3997 - val_loss: 31.6089
Epoch 40/50
 - 2s - loss: 31.3907 - val_loss: 31.1524
Epoch 41/50
 - 2s - loss: 31.3768 - val_loss: 31.4560
Epoch 42/50
 - 2s - loss: 31.3959 - val_loss: 31.4471
Epoch 43/50
 - 1s - loss: 31.4039 - val_loss: 31.2623
Epoch 44/50
 - 1s - loss: 31.3893 - val_loss: 31.3181
Epoch 45/50
 - 3s - loss: 31.3993 - val_loss: 31.3275
Epoch 46/50
 - 3s - loss: 31.3691 - val_loss: 31.6102
Epoch 47/50
 - 1s - loss: 31.4034 - val_loss: 31.5669
Epoch 48/50
 - 1s - loss: 31.4138 - val_loss: 31.3679
Epoch 49/50
 - 1s - loss: 31.4240 - val_loss: 31.3708
Epoch 50/50
 - 1s - loss: 31.3484 - val_loss: 31.3046
<keras.callbacks.callbacks.History object at 0x000001AC91DC8F28>
>>> complex_mean_absolute_error(y_test,model.predict(x_test))
<tf.Tensor: id=81329, shape=(), dtype=float64, numpy=0.09089791588512613>
>>> y_pred=model.predict(x_test)
>>> y_pred[:10]
array([[[ 6.01971118e-01, -3.01507790e-03]],

       [[ 2.48551018e-01,  1.23183947e-03]],

       [[ 9.71942274e-01,  3.59169866e-03]],

       [[ 2.99552179e-01,  1.86358824e-03]],

       [[ 4.28866875e-01,  3.19729820e-03]],

       [[ 1.84536579e+00,  7.22103286e-03]],

       [[ 1.50917908e-01,  4.14254300e-04]],

       [[ 2.80694849e-01,  1.67120131e-03]],

       [[ 3.58615420e-01,  2.49832427e-03]],

       [[ 1.86389035e+00,  7.29357333e-03]]])
>>> y_test[:1-]
SyntaxError: invalid syntax
>>> y_test[:10]
array([[0.28330834],
       [0.05996409],
       [1.42912851],
       [0.34600961],
       [0.04440514],
       [2.04711711],
       [0.42342132],
       [0.40800204],
       [0.39799744],
       [1.9326937 ]])
>>> y_pred[:10,0,0]
array([0.60197112, 0.24855102, 0.97194227, 0.29955218, 0.42886687,
       1.84536579, 0.15091791, 0.28069485, 0.35861542, 1.86389035])
>>> [reLogInv(weights) for weights in model.get_weights()]
[<tf.Tensor: id=81535, shape=(4, 8), dtype=float32, numpy=
array([[-1.2849588e-03, -1.8562368e-03, -2.1755395e-03, -3.1333435e-03,
        -3.1391245e-03,  1.0524655e-03, -1.7029910e-03,  3.0641851e-03],
       [-1.6910731e-03,  3.1163373e-03,  6.1318785e-04, -3.1246424e-03,
         2.7730428e-03,  3.1410139e-03, -6.9962459e-04,  2.8133306e-03],
       [-3.0263844e-03, -3.1398372e-03,  4.9744081e-04, -2.3578533e-03,
        -8.2271383e-04,  2.6095258e-03,  1.0015655e+00, -7.1137986e-04],
       [ 1.6184431e-03, -2.4714235e-03, -1.8219007e-03,  3.0408177e-04,
        -3.0414606e-03,  1.8506835e-05, -1.5301310e-04, -3.1208724e-03]],
      dtype=float32)>, <tf.Tensor: id=81543, shape=(8, 8), dtype=float32, numpy=
array([[-6.8741181e-04,  2.0914564e-03, -2.5770536e-03,  2.1022635e-03,
        -4.8011250e-05,  3.0769485e-03,  7.5143471e-04, -1.7431221e-03],
       [ 3.0067780e-03,  1.8767226e-03,  2.4661866e-03,  5.2809913e-04,
        -2.3270687e-03, -1.3734737e-03,  2.2908777e-03,  3.1184973e-03],
       [ 2.5768545e-03, -1.5696394e-03, -2.5518525e-03,  2.0046639e-03,
         1.1520502e-03,  1.7052139e-03,  2.5568127e-03, -4.1219010e-04],
       [-1.4309756e-03,  1.9304907e-03,  2.4273146e-03, -4.5376926e-04,
        -2.1702930e-04, -3.0205725e-03, -1.3891433e-03,  4.6688190e-04],
       [-3.0631907e-03, -2.1946605e-04, -5.8004010e-04,  3.0894647e-03,
        -1.8250938e-03,  1.8790547e-03, -3.0955432e-03,  1.8423174e-03],
       [ 7.5957074e-04, -3.0807580e-03,  2.7475492e-03,  2.2151812e-03,
        -9.9216378e-04, -1.3224740e-03, -2.8874360e-03,  1.3663776e-03],
       [-5.5343495e-04, -3.1508881e-04,  8.3377201e-04,  3.0472917e-03,
         2.7825620e-03, -9.3689247e-04,  3.8879373e-04, -1.1427408e+00],
       [ 3.1373056e-03,  2.7326907e-03, -8.9975784e-04,  1.9012147e-03,
        -1.8330248e-03, -7.8899309e-04, -2.4785071e-03,  1.4629686e-03]],
      dtype=float32)>, <tf.Tensor: id=81551, shape=(16, 8), dtype=float32, numpy=
array([[ 2.8428833e-03,  1.2193667e-04, -2.3005521e-03, -2.8342672e-03,
        -1.9192699e-03,  3.2910903e-04, -2.7552946e-03, -1.6668616e-03],
       [-2.8478787e-03,  1.8178885e-03, -7.3383615e-04, -8.0830534e-05,
        -2.0324008e-04, -1.2429780e-04, -6.1587023e-04, -1.1874433e-03],
       [ 2.4619566e-03, -2.0073198e-03,  2.5061911e-03, -1.9778556e-04,
         1.9276156e-03,  3.0438665e-03,  2.5632137e-03,  4.4820618e-04],
       [-1.8905982e-03, -1.5502248e-03, -2.1159125e-03,  1.3473653e-04,
        -3.0036692e-03, -2.9764243e-03,  2.4069706e-03,  3.2096976e-04],
       [-2.7952262e-03,  2.9211631e-04,  1.8316867e-03,  7.4363372e-05,
         3.1147148e-03,  1.0184374e-03,  1.4072071e-03, -2.0513807e-03],
       [ 2.9657953e-03,  1.8727034e-05,  3.0600571e-03,  1.2262132e-03,
        -1.2816322e-03,  6.8487693e-04, -1.3559615e-03, -4.7288899e-04],
       [-2.1355134e-03,  1.5886474e-03, -2.7150949e-03,  2.2949958e-03,
         2.4299035e-03, -2.7116972e-03,  1.0255361e-03, -3.1448842e-05],
       [-7.8907411e-05,  1.3592860e-03,  1.7008748e-03,  3.0602692e-03,
        -1.2759153e+00, -2.5248483e-03,  1.3885559e-03, -2.1584975e-03],
       [-1.1859515e-03, -2.8965585e-03,  3.0924056e-03, -2.7344020e-03,
        -2.6537233e-03, -2.6744853e-03,  3.1178896e-03, -1.8189688e-03],
       [-1.8750099e-03, -9.9476811e-04,  2.5344610e-03, -2.2513734e-05,
        -2.8299049e-03, -2.0494647e-03,  2.7769608e-03,  2.0205139e-03],
       [-1.6138704e-03, -2.7421631e-03, -4.5974550e-04, -3.0558133e-03,
         2.5442187e-03,  2.9345709e-03,  1.1528771e-03,  3.0624922e-03],
       [-2.4244664e-03,  2.5888761e-03, -4.1976175e-04,  1.9957104e-03,
        -2.3023000e-03,  6.9077208e-04,  2.7028844e-05, -1.2922455e-03],
       [-1.6680409e-03, -1.5068625e-04,  9.9033886e-04, -2.6443431e-03,
         1.6035101e-03,  1.5644134e-03, -2.1762543e-03,  2.0325710e-03],
       [-1.4705035e-03, -5.5499061e-04, -2.9555969e-03, -1.9353048e-03,
         2.8522427e-03, -1.5541838e-03,  3.1577970e-03, -8.2512293e-04],
       [ 1.5949509e-03,  1.3512208e-03, -2.5737768e-03,  2.7574173e-03,
        -1.0190895e-03, -9.3928224e-04,  2.0349631e-03, -7.4501120e-04],
       [-1.0407397e-03,  1.7865512e-03, -1.7070438e-04,  1.0743132e-03,
        -2.1902069e-03,  2.4350560e-03, -1.3961244e-03, -3.9181398e-04]],
      dtype=float32)>, <tf.Tensor: id=81559, shape=(8, 1), dtype=float32, numpy=
array([[ 0.00129132],
       [ 0.00055011],
       [ 0.00066229],
       [-0.00289042],
       [ 0.53276896],
       [ 0.00299698],
       [-0.0022541 ],
       [-0.00072977]], dtype=float32)>]
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 137, in <module>
    p_dense_1 = PowerDense(8, True, copy_log = True, name='power_dense_1', kernel_regularizer=llog)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 463, in __call__
    self.build(unpack_singleton(input_shapes))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 90, in build
    constraint=self.kernel_constraint)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 285, in add_weight
    self.add_loss(regularizer(weight))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 105, in llog
    return 0.01*tf.cast(tf.reduce_sum(values),tf.float64)
NameError: name 'values' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: 46.5931 - val_loss: 49.0897
Epoch 2/50
 - 2s - loss: 41.6124 - val_loss: 40.4729
Epoch 3/50
 - 2s - loss: 39.5157 - val_loss: 38.4436
Epoch 4/50
 - 2s - loss: 37.9465 - val_loss: 37.5696
Epoch 5/50
 - 1s - loss: 37.0678 - val_loss: 36.6356
Epoch 6/50
 - 3s - loss: 36.3117 - val_loss: 36.1960
Epoch 7/50
 - 2s - loss: 35.5412 - val_loss: 35.1725
Epoch 8/50
 - 2s - loss: 34.5828 - val_loss: 34.1538
Epoch 9/50
 - 2s - loss: 34.1646 - val_loss: 34.0455
Epoch 10/50
 - 1s - loss: 33.9851 - val_loss: 33.9602
Epoch 11/50
 - 1s - loss: 33.9107 - val_loss: 33.6920
Epoch 12/50
 - 1s - loss: 33.9194 - val_loss: 33.8900
Epoch 13/50
 - 2s - loss: 33.8710 - val_loss: 34.0449
Epoch 14/50
 - 2s - loss: 33.8812 - val_loss: 33.8442
Epoch 15/50
 - 2s - loss: 33.8397 - val_loss: 33.9842
Epoch 16/50
 - 2s - loss: 33.8735 - val_loss: 33.8994
Epoch 17/50
 - 2s - loss: 33.8482 - val_loss: 34.0386
Epoch 18/50
 - 2s - loss: 33.8578 - val_loss: 33.7140
Epoch 19/50
 - 2s - loss: 33.8825 - val_loss: 33.8378
Epoch 20/50
 - 2s - loss: 33.8685 - val_loss: 33.9382
Epoch 21/50
 - 2s - loss: 33.8881 - val_loss: 33.6589
Epoch 22/50
 - 2s - loss: 33.7461 - val_loss: 33.7619
Epoch 23/50
 - 2s - loss: 33.7231 - val_loss: 33.4851
Epoch 24/50
 - 2s - loss: 33.7051 - val_loss: 33.4376
Epoch 25/50
 - 2s - loss: 33.6454 - val_loss: 33.5026
Epoch 26/50
 - 2s - loss: 33.5749 - val_loss: 33.2649
Epoch 27/50
 - 2s - loss: 33.5651 - val_loss: 33.5730
Epoch 28/50
 - 3s - loss: 33.5382 - val_loss: 33.3550
Epoch 29/50
 - 3s - loss: 33.5683 - val_loss: 33.1958
Epoch 30/50
 - 2s - loss: 33.5931 - val_loss: 33.3040
Epoch 31/50
 - 2s - loss: 33.5297 - val_loss: 33.3451
Epoch 32/50
 - 2s - loss: 33.4985 - val_loss: 33.7231
Epoch 33/50
 - 2s - loss: 33.5368 - val_loss: 33.6940
Epoch 34/50
 - 2s - loss: 33.5185 - val_loss: 33.6164
Epoch 35/50
 - 2s - loss: 33.5174 - val_loss: 33.6056
Epoch 36/50
 - 1s - loss: 33.4832 - val_loss: 33.6544
Epoch 37/50
 - 1s - loss: 33.4680 - val_loss: 33.5503
Epoch 38/50
 - 1s - loss: 33.5099 - val_loss: 33.5010
Epoch 39/50
 - 3s - loss: 33.4708 - val_loss: 33.5464
Epoch 40/50
 - 3s - loss: 33.4514 - val_loss: 33.4435
Epoch 41/50
 - 2s - loss: 33.4959 - val_loss: 33.6859
Epoch 42/50
 - 2s - loss: 33.4556 - val_loss: 33.1235
Epoch 43/50
 - 2s - loss: 33.4468 - val_loss: 33.5319
Epoch 44/50
 - 2s - loss: 33.4258 - val_loss: 33.2913
Epoch 45/50
 - 1s - loss: 33.4148 - val_loss: 33.4312
Epoch 46/50
 - 1s - loss: 33.4061 - val_loss: 33.5340
Epoch 47/50
 - 1s - loss: 33.4337 - val_loss: 33.6863
Epoch 48/50
 - 1s - loss: 33.4503 - val_loss: 33.4211
Epoch 49/50
 - 3s - loss: 33.4193 - val_loss: 33.6628
Epoch 50/50
 - 2s - loss: 33.4526 - val_loss: 33.4764
>>> complex_mean_absolute_error(y_test,model.predict(x_test))
<tf.Tensor: id=81473, shape=(), dtype=float64, numpy=0.12610230886253732>
>>> [reLogInv(weights) for weights in model.get_weights()]
[<tf.Tensor: id=81490, shape=(4, 8), dtype=float32, numpy=
array([[-2.26710830e-03,  3.11370869e-03,  2.31850124e-03,
         3.11428308e-03, -1.75351487e-03,  1.31858443e-03,
         1.87069306e-03, -3.05752561e-04],
       [-1.28209265e-03, -2.11462239e-03,  2.97074392e-03,
        -6.31779199e-04, -1.46007014e-03,  2.34084530e-03,
        -1.35939685e-03, -2.60067149e-03],
       [-2.64578243e-03,  2.21467833e-03,  9.85767126e-01,
        -3.04566696e-03,  4.26603365e-05, -2.70480174e-03,
         3.15569993e-03, -4.47384315e-04],
       [-2.71755969e-03,  3.13752005e-03,  1.12971314e-03,
         9.18409089e-04,  1.04390085e-04, -7.21285702e-04,
         2.54159654e-03, -2.99352151e-03]], dtype=float32)>, <tf.Tensor: id=81498, shape=(8, 8), dtype=float32, numpy=
array([[-3.1159350e-03, -7.2153029e-04, -7.8518025e-04, -3.1220587e-03,
        -2.6149254e-03,  2.5010959e-03, -2.4090903e-03, -2.6604859e-03],
       [ 2.2691437e-03, -1.5913112e-03, -2.8253123e-03,  1.1389324e-03,
         2.5355013e-03,  0.0000000e+00, -2.8473331e-04,  1.6613955e-03],
       [ 2.8660116e-03, -5.8659283e-04, -6.3247455e-05, -4.4135544e-03,
         3.3262111e-02, -1.9239773e-03,  6.3173752e-04, -3.6316412e-04],
       [-3.9950223e-04,  2.6839687e-03, -1.6029194e-03, -1.7252903e-03,
        -4.3923255e-02,  5.0541526e-04, -3.0471880e-03, -3.0550675e-03],
       [ 2.5281061e-03,  2.6502598e-03, -3.6193139e-04,  1.5689699e-03,
         1.5601838e-03,  1.5542152e-03,  1.7658026e-03, -1.7195824e-03],
       [ 2.3373629e-03, -1.9794500e-03,  1.3830800e-03, -3.2073488e-03,
         2.3282308e-03, -7.2492438e-04,  1.2019074e-03,  2.9141901e-03],
       [-5.6570047e-04,  2.4829770e-03, -8.4656430e-04, -2.6066969e-03,
        -2.5076922e-03,  3.5093268e-04, -8.2975323e-04, -1.2022544e-03],
       [ 2.9219780e-04,  5.0136528e-05, -1.0999511e-03, -1.7279868e-03,
         1.1085550e-03, -2.3716446e-03, -1.1293216e-03,  3.1420297e-03]],
      dtype=float32)>, <tf.Tensor: id=81506, shape=(16, 8), dtype=float32, numpy=
array([[ 3.1403101e-03, -7.1175763e-04,  1.5026757e-03,  1.5364876e-03,
        -1.3973271e-03, -3.0988681e-03, -1.5731116e-03,  3.0857197e-03],
       [-1.6520065e-03, -4.3165960e-04, -2.8161192e-03,  2.3298590e-03,
        -1.2049695e-03,  2.9402210e-03,  7.0172723e-04,  3.1561023e-03],
       [-1.7169956e-04, -2.0106353e-03,  2.4628921e-03,  2.1642216e-03,
        -6.2669121e-04, -3.3373130e-05, -9.7934192e-04, -2.7388164e-03],
       [ 1.3436314e-03,  3.0839413e-03, -1.9658050e-03,  9.6495962e-05,
         8.8683807e-04,  1.7028865e-03,  6.8144064e-04,  2.8440680e-03],
       [-2.0527234e-03,  1.5423386e-03, -1.7402405e-03, -2.6372673e-03,
         2.7106677e-03,  3.1328257e-04,  1.1906087e-03, -7.6993997e-04],
       [ 3.0112425e-03, -3.1358437e-03,  2.5222974e-04, -2.3701622e-03,
        -2.2623308e-03, -7.7680650e-04, -2.9145693e-03,  2.4266553e-03],
       [-1.8311398e-03,  2.0582958e-03,  2.0658432e-03,  2.6879101e-03,
         3.0609192e-03, -1.6891570e-03, -2.6851422e-03,  3.0072988e-05],
       [ 6.2312046e-04,  2.2000519e-03,  1.8242009e-03, -1.4305926e-03,
        -6.7702914e-04,  2.7995373e-03,  1.4230901e-03, -1.5272687e-03],
       [-9.2349877e-04,  2.6572142e-03,  1.4425811e-03,  2.8433211e-03,
        -9.0024318e-05,  2.9040743e-03,  1.2158351e-03, -2.7472903e-03],
       [ 2.0862638e-03, -2.1157553e-03, -1.8530102e-03, -2.6503126e-03,
        -1.9875218e-03,  2.3168698e-03, -2.1213959e-03, -1.1838973e-03],
       [-3.0830314e-03, -3.0575336e-03, -7.5668650e-04, -1.4269010e-03,
        -2.0456156e-03, -3.0903909e-03,  1.7272581e-03,  2.8177822e-04],
       [-1.3661950e-03, -1.9143329e-03, -2.9350063e-03,  9.2526100e-04,
         1.5565952e-03,  1.3626213e-03, -1.2313443e-03,  1.7988584e-03],
       [ 1.4434988e-03, -1.0009651e+00,  1.4432141e-04, -1.0122096e-03,
         2.8442771e-03,  0.0000000e+00,  2.4410123e-03, -1.4887600e-03],
       [ 2.3218873e-04,  2.4156438e-03, -5.3329283e-04,  2.9109537e-03,
         7.8686979e-04,  1.7488848e-03, -2.7568045e-03,  3.0408744e-03],
       [ 1.8520008e-03, -5.7784433e-05, -3.1220899e-03,  3.0686946e-03,
        -8.4474223e-04,  2.7409729e-03, -5.8202911e-04, -2.9023250e-03],
       [-2.2559026e-03, -8.1065414e-04,  2.6990026e-03, -1.0080610e-03,
        -8.7472436e-04, -2.9123558e-03,  2.2197145e-03, -4.2645581e-04]],
      dtype=float32)>, <tf.Tensor: id=81514, shape=(8, 1), dtype=float32, numpy=
array([[-8.7535067e-05],
       [-2.5381073e-02],
       [-1.0107774e-03],
       [ 2.0643256e-03],
       [-1.3806645e-03],
       [-2.7872336e-03],
       [-2.6286193e-03],
       [-2.5629392e-03]], dtype=float32)>]
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 6s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 155, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> y_pred = model.predict(x_train)
>>> loss = complex_absolute_error(y_train, y_pred)
>>> tf.reduce_sum(loss)
<tf.Tensor: id=3752, shape=(), dtype=float64, numpy=1.1395944178115944e+31>
>>> tf.cast(tf.reduce_sum(loss),tf.float32)
<tf.Tensor: id=3756, shape=(), dtype=float32, numpy=1.1395944e+31>
>>> tf.cast(tf.reduce_sum(loss)*tf.reduce_sum(loss),tf.float32)
<tf.Tensor: id=3763, shape=(), dtype=float32, numpy=inf>
>>> tf.cast(tf.reduce_sum(loss)*10000,tf.float32)
<tf.Tensor: id=3769, shape=(), dtype=float32, numpy=1.1395944e+35>
>>> tf.cast(tf.reduce_sum(loss)*10**10,tf.float32)
<tf.Tensor: id=3775, shape=(), dtype=float32, numpy=inf>
>>> np.exp(70)
2.515438670919167e+30
>>> 2**64
18446744073709551616
>>> np.exp(10)
22026.465794806718
>>> loss.shape
TensorShape([8000, 1, 2])
>>> lossadj = loss/loss.shape[0]
>>> tf.reduce_sum(lossadj)
<tf.Tensor: id=3780, shape=(), dtype=float64, numpy=1.4244930222644932e+27>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 143, in <module>
    model.compile(loss=complex_absolute_error,optimizer="rmsprop", metrics=[])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 229, in compile
    self.total_loss = self._prepare_total_loss(masks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 692, in _prepare_total_loss
    y_true, y_pred, sample_weight=sample_weight)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 71, in __call__
    losses = self.call(y_true, y_pred)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\losses.py", line 132, in call
    return self.fn(y_true, y_pred, **self._fn_kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 118, in complex_absolute_error
    return tf.cast(tf.abs(y_pred - y_true)/y_true.shape[0],tf.float64)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 888, in binary_op_wrapper
    y, dtype_hint=x.dtype.base_dtype, name="y")
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 284, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 455, in make_tensor_proto
    raise ValueError("None values not supported.")
ValueError: None values not supported.
>>> y_pred = model.preict(x_train)
Traceback (most recent call last):
  File "<pyshell#584>", line 1, in <module>
    y_pred = model.preict(x_train)
AttributeError: 'Model' object has no attribute 'preict'
>>> y_pred = model.predict(x_train)
Traceback (most recent call last):
  File "<pyshell#585>", line 1, in <module>
    y_pred = model.predict(x_train)
NameError: name 'x_train' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 6s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "<pyshell#586>", line 1, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> kernel = model.get_weights()
>>> l0_1(kernel)
Traceback (most recent call last):
  File "<pyshell#588>", line 1, in <module>
    l0_1(kernel)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 94, in l0_1
    return tf.cast(tf.pow(tf.norm(weight_matrix,ord=0.01),0.01),tf.float64)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\linalg_ops.py", line 493, in norm_v2
    name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\linalg_ops.py", line 593, in norm
    tensor = ops.convert_to_tensor(tensor)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1100, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 254, in _constant_impl
    t = convert_to_eager_tensor(value, ctx, dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 115, in convert_to_eager_tensor
    return ops.EagerTensor(value, handle, device, dtype)
ValueError: Can't convert non-rectangular Python sequence to Tensor.
>>> l0_1(kernel[0])
<tf.Tensor: id=5236, shape=(), dtype=float64, numpy=nan>
>>> l0_1(kernel[1])
<tf.Tensor: id=5249, shape=(), dtype=float64, numpy=nan>
>>> kernel[0]
array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> kernel = model.get_weights()
>>> l0_1(kernel[0])
<tf.Tensor: id=2846, shape=(), dtype=float64, numpy=inf>
>>> kernel[0]
array([[-0.03066927,  0.6031423 , -0.1679225 ,  0.26671875, -0.41689268,
         0.61644524,  0.34114045,  0.2672766 ],
       [-0.5456269 ,  0.38698238, -0.19375765, -0.04071403,  0.03650016,
        -0.5574293 , -0.28886285, -0.28175405],
       [ 0.59362286, -0.53718853, -0.15141559,  0.25239134, -0.6610195 ,
        -0.04817873, -0.03806567,  0.10732591],
       [-0.54596704,  0.28880286, -0.08641756, -0.5891605 , -0.5822245 ,
        -0.329032  , -0.46045414,  0.6413012 ]], dtype=float32)
>>> tf.norm(kernel[0],ord=0.01)
<tf.Tensor: id=2856, shape=(), dtype=float32, numpy=inf>
>>> tf.abs(kernel[0])
<tf.Tensor: id=2859, shape=(4, 8), dtype=float32, numpy=
array([[0.03066927, 0.6031423 , 0.1679225 , 0.26671875, 0.41689268,
        0.61644524, 0.34114045, 0.2672766 ],
       [0.5456269 , 0.38698238, 0.19375765, 0.04071403, 0.03650016,
        0.5574293 , 0.28886285, 0.28175405],
       [0.59362286, 0.53718853, 0.15141559, 0.25239134, 0.6610195 ,
        0.04817873, 0.03806567, 0.10732591],
       [0.54596704, 0.28880286, 0.08641756, 0.5891605 , 0.5822245 ,
        0.329032  , 0.46045414, 0.6413012 ]], dtype=float32)>
>>> tf.pow(tf.abs(kernel[0]),0.01)
<tf.Tensor: id=2864, shape=(4, 8), dtype=float32, numpy=
array([[0.96575516, 0.99495673, 0.9823157 , 0.98687136, 0.9912889 ,
        0.9951738 , 0.989303  , 0.986892  ],
       [0.99396014, 0.9905512 , 0.98372245, 0.96849513, 0.96743757,
        0.9941729 , 0.98765874, 0.9874127 ],
       [0.9947985 , 0.9938052 , 0.9812998 , 0.9863266 , 0.99586886,
        0.9701269 , 0.96784395, 0.9779284 ],
       [0.9939663 , 0.9876567 , 0.9758117 , 0.9947234 , 0.9946056 ,
        0.98894554, 0.9922746 , 0.9955673 ]], dtype=float32)>
>>> tf.reduce_sum(tf.pow(tf.abs(kernel[0]),0.01))
<tf.Tensor: id=2871, shape=(), dtype=float32, numpy=31.557518>
>>> tf.reduce_mean(tf.pow(tf.abs(kernel[0]),0.01))
<tf.Tensor: id=2878, shape=(), dtype=float32, numpy=0.98617244>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 138, in <module>
    p_dense_1 = PowerDense(8, True, copy_log = True, name='power_dense_1', kernel_regularizer=l0_1)(value_in)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 463, in __call__
    self.build(unpack_singleton(input_shapes))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 90, in build
    constraint=self.kernel_constraint)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 285, in add_weight
    self.add_loss(regularizer(weight))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 95, in l0_1
    tf.reduce_mean(tf.pow(tf.abs(kernel[0]),0.01))
NameError: name 'kernel' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "<pyshell#600>", line 1, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> y_pred = model.predict(x_train)
>>> loss = complex_absolute_error(y_train,y_pred)
>>> reg = [l0_1(weights) for weights in model.get_weights()]
>>> reg.shape
Traceback (most recent call last):
  File "<pyshell#604>", line 1, in <module>
    reg.shape
AttributeError: 'list' object has no attribute 'shape'
>>> np.sum(reg)
Traceback (most recent call last):
  File "<pyshell#605>", line 1, in <module>
    np.sum(reg)
  File "<__array_function__ internals>", line 6, in sum
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\fromnumeric.py", line 2182, in sum
    initial=initial, where=where)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\fromnumeric.py", line 90, in _wrapreduction
    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'
>>> reg
[None, None, None, None]
>>> model.get_weights()
[array([[-0.59331834,  0.567282  , -0.34959394, -0.08453375,  0.5334855 ,
        -0.4450016 , -0.58348346,  0.25008255],
       [ 0.41579884,  0.36887532, -0.30182537,  0.19417763,  0.53290707,
         0.6419149 , -0.34243503, -0.6120719 ],
       [-0.20132017,  0.41729134, -0.20498407, -0.20183283,  0.4540817 ,
         0.17580414, -0.37463525,  0.4869377 ],
       [-0.21250644,  0.3366515 , -0.2857639 , -0.47952878, -0.36703986,
         0.3584743 ,  0.16455191,  0.3503216 ]], dtype=float32), array([[ 0.08666605,  0.28336805,  0.09708595,  0.06570029,  0.56246156,
         0.28927803,  0.04690033,  0.0318777 ],
       [-0.0256353 , -0.12043458, -0.6069897 ,  0.36855447,  0.33756608,
        -0.08590204, -0.32769275,  0.2800662 ],
       [ 0.00075716, -0.06020921, -0.4009254 ,  0.53424734,  0.11997676,
         0.19313139,  0.45619756, -0.17756668],
       [ 0.11345285, -0.05411309,  0.44273728,  0.4025826 , -0.29070967,
        -0.19913056,  0.46383482,  0.52333826],
       [-0.10214138, -0.371849  ,  0.58545715,  0.59184045,  0.19016892,
        -0.15917084,  0.08103782, -0.16716439],
       [ 0.3976726 , -0.552946  , -0.59096885,  0.33152044,  0.45984906,
         0.3867995 , -0.4195757 , -0.04715687],
       [-0.19401908,  0.3920341 , -0.42187566,  0.267133  ,  0.3517952 ,
         0.4544391 ,  0.35152608, -0.14868301],
       [-0.4687599 , -0.582662  , -0.60509944,  0.3736229 ,  0.05850536,
        -0.27505326,  0.1308493 , -0.2851713 ]], dtype=float32), array([[ 0.11006176,  0.29011345, -0.0440681 , -0.26660335, -0.17816007,
         0.3705244 , -0.4670819 , -0.10082245],
       [ 0.15031469,  0.35280526,  0.09278893,  0.36343908, -0.48016143,
        -0.1662736 ,  0.12008905,  0.13496745],
       [ 0.15300655, -0.24053109, -0.46119082,  0.26928854,  0.33134818,
        -0.1723175 ,  0.27875638,  0.47661448],
       [ 0.2626047 , -0.4322394 , -0.32220817, -0.48856568, -0.28595757,
        -0.29039907,  0.28335536, -0.2271775 ],
       [ 0.08447945,  0.3304988 , -0.09786952,  0.216954  , -0.24578702,
         0.23328447,  0.41790557,  0.27043474],
       [-0.32913136, -0.44546008,  0.2449193 , -0.20846224, -0.3670243 ,
        -0.04672933,  0.20122123, -0.4964819 ],
       [-0.19465351, -0.17185318,  0.01131976,  0.35450935,  0.14150548,
         0.22377408, -0.08861661, -0.12288082],
       [-0.11898303,  0.39126766, -0.07953906, -0.31420112, -0.03277659,
        -0.30880737,  0.34795904, -0.25013435],
       [-0.31342983,  0.05056608,  0.4494524 , -0.38298023,  0.3397658 ,
         0.18256426, -0.49936318, -0.06067884],
       [ 0.28287923, -0.0996201 ,  0.4925661 ,  0.15991807,  0.08615923,
         0.1804583 ,  0.46234798,  0.04753995],
       [ 0.43044174,  0.26614356,  0.48898935,  0.30253613,  0.1263895 ,
        -0.09042978, -0.05036962, -0.31548142],
       [-0.34986746, -0.44900692,  0.00859034, -0.2737099 , -0.49349785,
        -0.4024576 , -0.32561827,  0.12755358],
       [-0.32596612, -0.40658855,  0.21517467, -0.49445915, -0.36012578,
         0.47785735,  0.2609775 , -0.05590689],
       [-0.09873843, -0.41439545,  0.105304  , -0.4738536 ,  0.09640086,
        -0.2796651 ,  0.43840408,  0.16951573],
       [-0.06667459,  0.47737145,  0.35708904,  0.3646661 ,  0.424178  ,
        -0.18743634,  0.4853691 , -0.43848348],
       [-0.11207557, -0.4173106 ,  0.16429865, -0.4203434 , -0.03190804,
        -0.34138095, -0.3249954 ,  0.3421322 ]], dtype=float32), array([[-0.505249  ],
       [-0.27487564],
       [-0.41242996],
       [ 0.35343528],
       [ 0.8159934 ],
       [ 0.3318764 ],
       [ 0.07264096],
       [ 0.77968025]], dtype=float32)]
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 527, in _apply_op_helper
    preferred_dtype=default_dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1036, in _TensorTensorConversionFunction
    (dtype.name, t.dtype.name, str(t)))
ValueError: Tensor conversion requested dtype float64 for Tensor with dtype float32: 'Tensor("power_dense_1/weight_regularizer/Mean:0", shape=(), dtype=float32)'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 144, in <module>
    model.compile(loss=complex_absolute_error,optimizer="rmsprop", metrics=[])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 229, in compile
    self.total_loss = self._prepare_total_loss(masks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 713, in _prepare_total_loss
    total_loss += loss_tensor
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 884, in binary_op_wrapper
    return func(x, y, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 396, in add
    "Add", x=x, y=y, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 563, in _apply_op_helper
    inferred_from[input_arg.type_attr]))
TypeError: Input 'y' of 'Add' Op has type float32 that does not match type float64 of argument 'x'.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
Traceback (most recent call last):
  File "<pyshell#608>", line 1, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> 
>>> y_pred = model.predict(x_train)
>>> loss = complex_absolute_error(y_train,y_pred)
>>> reg = [l0_1(weights) for weights in model.get_weights()]
>>> reg
[<tf.Tensor: id=3425, shape=(), dtype=float64, numpy=0.9864463806152344>, <tf.Tensor: id=3432, shape=(), dtype=float64, numpy=0.9841838479042053>, <tf.Tensor: id=3439, shape=(), dtype=float64, numpy=0.9846807718276978>, <tf.Tensor: id=3446, shape=(), dtype=float64, numpy=0.9868479371070862>]
>>> loss.shape
TensorShape([8000, 1, 2])
>>> loss[:10]
<tf.Tensor: id=3454, shape=(10, 1, 2), dtype=float64, numpy=
array([[[ 0.57241579,  2.477465  ]],

       [[11.88330179, 25.42982531]],

       [[ 3.65603082,  0.72408056]],

       [[ 5.96574717,  5.96803268]],

       [[28.54081255, 47.36017624]],

       [[ 0.63872585,  2.17500823]],

       [[ 5.07627913,  5.09009551]],

       [[ 0.75440126,  2.14897282]],

       [[ 0.29348158,  3.20506974]],

       [[ 0.51918585,  0.32282464]]])>
>>> tf.reduce_sum(loss)+sum(reg)
<tf.Tensor: id=3463, shape=(), dtype=float64, numpy=4.8196584678028616e+30>
>>> tf.cast(tf.reduce_sum(loss)+sum(reg),tf.float32)
<tf.Tensor: id=3473, shape=(), dtype=float32, numpy=4.8196584e+30>
>>> tf.cast(tf.reduce_sum(loss)+sum(reg),tf.float16)
<tf.Tensor: id=3483, shape=(), dtype=float16, numpy=inf>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 5s - loss: nan - val_loss: nan
<keras.callbacks.callbacks.History object at 0x000002338D741940>
>>> np.exp(20)
485165195.4097903
>>> tf.max(x_test)
Traceback (most recent call last):
  File "<pyshell#621>", line 1, in <module>
    tf.max(x_test)
AttributeError: module 'tensorflow' has no attribute 'max'
>>> tf.maximum(x_test)
Traceback (most recent call last):
  File "<pyshell#622>", line 1, in <module>
    tf.maximum(x_test)
TypeError: maximum() missing 1 required positional argument: 'y'
>>> tf.reduce_max(x_test)
<tf.Tensor: id=4058, shape=(), dtype=float64, numpy=0.9996460629019421>
>>> tf.reduce_max(y_test)
<tf.Tensor: id=4066, shape=(), dtype=float64, numpy=2.4472130221467943>
>>> tf.reduce_min(y_test)
<tf.Tensor: id=4074, shape=(), dtype=float64, numpy=-0.0037154741131144586>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 5s - loss: 774846.6630 - val_loss: 826381.4912
<keras.callbacks.callbacks.History object at 0x00000254E2F81940>
>>> model.loss
<function complex_absolute_error at 0x00000254E1316488>
>>> loss=model.losses
>>> loss.shape
Traceback (most recent call last):
  File "<pyshell#629>", line 1, in <module>
    loss.shape
AttributeError: 'list' object has no attribute 'shape'
>>> len(loss)
4
>>> loss
[<tf.Tensor 'power_dense_1/weight_regularizer/Cast:0' shape=() dtype=float64>, <tf.Tensor 'dense_1/weight_regularizer/Cast:0' shape=() dtype=float64>, <tf.Tensor 'power_dense_2/weight_regularizer/Cast:0' shape=() dtype=float64>, <tf.Tensor 'dense_2/weight_regularizer/Cast:0' shape=() dtype=float64>]
>>> loss[0].shape
TensorShape([])
>>> loss[1].shape
TensorShape([])
>>> loss[0]
<tf.Tensor 'power_dense_1/weight_regularizer/Cast:0' shape=() dtype=float64>
>>> loss[1]
<tf.Tensor 'dense_1/weight_regularizer/Cast:0' shape=() dtype=float64>
>>> loss[0].eval()
Traceback (most recent call last):
  File "<pyshell#636>", line 1, in <module>
    loss[0].eval()
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 739, in eval
    return _eval_using_default_session(self, feed_dict, self.graph, session)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 5245, in _eval_using_default_session
    raise ValueError("Cannot evaluate tensor using `eval()`: No default "
ValueError: Cannot evaluate tensor using `eval()`: No default session is registered. Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 5s - loss: 18798804325.3823 - val_loss: 9627347037.5697
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 5s - loss: 148638628005024.5938 - val_loss: 184136335602205.6250
>>> np.exp(40)
2.3538526683702e+17
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 156, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 2s - loss: 89934992.6204 - val_loss: 213217.4361
Epoch 2/50
 - 1s - loss: 4416067.3156 - val_loss: 22433.7241
Epoch 3/50
 - 1s - loss: 348447.9850 - val_loss: 3526.9177
Epoch 4/50
 - 1s - loss: 29505.5273 - val_loss: 683.6851
Epoch 5/50
 - 1s - loss: 1359.1329 - val_loss: 261.7038
Epoch 6/50
 - 1s - loss: 311.2259 - val_loss: 243.6421
Epoch 7/50
 - 1s - loss: 237.7578 - val_loss: 225.2070
Epoch 8/50
 - 1s - loss: 223.6366 - val_loss: 223.5456
Epoch 9/50
 - 3s - loss: 222.9685 - val_loss: 222.7042
Epoch 10/50
 - 2s - loss: 222.0242 - val_loss: 221.2053
Epoch 11/50
 - 2s - loss: 220.3289 - val_loss: 219.8605
Epoch 12/50
 - 2s - loss: 219.3550 - val_loss: 219.1422
Epoch 13/50
 - 2s - loss: 218.9794 - val_loss: 218.7699
Epoch 14/50
 - 2s - loss: 218.5834 - val_loss: 218.1521
Epoch 15/50
 - 2s - loss: 218.2244 - val_loss: 217.8987
Epoch 16/50
 - 1s - loss: 218.1128 - val_loss: 218.2153
Epoch 17/50
 - 1s - loss: 218.0642 - val_loss: 218.2034
Epoch 18/50
 - 1s - loss: 218.0582 - val_loss: 218.0143
Epoch 19/50
 - 1s - loss: 217.9817 - val_loss: 217.9985
Epoch 20/50
 - 1s - loss: 217.8469 - val_loss: 217.7279
Epoch 21/50
 - 3s - loss: 217.8349 - val_loss: 217.6566
Epoch 22/50
 - 3s - loss: 217.8355 - val_loss: 218.0083
Epoch 23/50
 - 2s - loss: 217.8478 - val_loss: 217.7227
Epoch 24/50
 - 1s - loss: 217.8103 - val_loss: 217.9726
Epoch 25/50
 - 1s - loss: 217.7652 - val_loss: 217.7146
Epoch 26/50
 - 1s - loss: 217.7316 - val_loss: 217.7933
Epoch 27/50
 - 1s - loss: 217.7459 - val_loss: 217.5247
Epoch 28/50
 - 1s - loss: 217.7030 - val_loss: 217.7546
Epoch 29/50
 - 1s - loss: 217.7224 - val_loss: 217.6479
Epoch 30/50
 - 1s - loss: 217.6643 - val_loss: 217.6644
Epoch 31/50
 - 1s - loss: 217.6926 - val_loss: 217.5995
Epoch 32/50
 - 1s - loss: 217.6312 - val_loss: 217.5293
Epoch 33/50
 - 3s - loss: 217.6817 - val_loss: 217.5240
Epoch 34/50
 - 2s - loss: 217.6516 - val_loss: 217.8986
Epoch 35/50
 - 2s - loss: 217.6544 - val_loss: 217.8554
Epoch 36/50
 - 1s - loss: 217.6487 - val_loss: 217.7250
Epoch 37/50
 - 3s - loss: 217.6627 - val_loss: 217.7900
Epoch 38/50
 - 3s - loss: 217.6447 - val_loss: 217.7008
Epoch 39/50
 - 2s - loss: 217.6473 - val_loss: 217.7399
Epoch 40/50
 - 2s - loss: 217.5945 - val_loss: 217.5889
Epoch 41/50
 - 3s - loss: 217.6251 - val_loss: 217.7611
Epoch 42/50
 - 2s - loss: 217.6180 - val_loss: 217.7314
Epoch 43/50
 - 1s - loss: 217.5885 - val_loss: 217.7352
Epoch 44/50
 - 1s - loss: 217.6011 - val_loss: 217.5208
Epoch 45/50
 - 1s - loss: 217.5919 - val_loss: 217.4858
Epoch 46/50
 - 3s - loss: 217.6322 - val_loss: 217.7003
Epoch 47/50
 - 3s - loss: 217.6219 - val_loss: 217.6718
Epoch 48/50
 - 2s - loss: 217.6416 - val_loss: 217.7062
Epoch 49/50
 - 2s - loss: 217.6261 - val_loss: 217.6872
Epoch 50/50
 - 1s - loss: 217.6252 - val_loss: 217.8714
>>> complex_mean_absolute_error(y_test,model.predict(x_test))
<tf.Tensor: id=81085, shape=(), dtype=float64, numpy=0.1291376180268947>
>>> y_pred=model.predict(x_test)
>>> y_test[:10]
array([[ 2.80755716e-01],
       [ 1.72337442e+00],
       [ 1.14633175e+00],
       [-1.15365309e-03],
       [ 5.80583914e-01],
       [ 3.07634184e-01],
       [ 1.91067453e+00],
       [ 3.67363935e-01],
       [ 3.34145454e-02],
       [ 4.37087871e-01]])
>>> y_pred[:10]
array([[[ 0.28478737,  0.07218287]],

       [[ 1.82613001,  0.02708921]],

       [[ 0.65293924,  0.00786   ]],

       [[ 0.1912654 ,  0.01230729]],

       [[ 0.31614657,  0.07289431]],

       [[ 0.21313114,  0.04142069]],

       [[ 2.05736439, -0.22100096]],

       [[ 0.41425434,  0.00395289]],

       [[ 0.20269812,  0.02054913]],

       [[ 0.20916793,  0.03692715]]])
>>> y_test[:10,1]
Traceback (most recent call last):
  File "<pyshell#642>", line 1, in <module>
    y_test[:10,1]
IndexError: index 1 is out of bounds for axis 1 with size 1
>>> y_test.shape
(2000, 1)
>>> y_test[:10,0]
array([ 2.80755716e-01,  1.72337442e+00,  1.14633175e+00, -1.15365309e-03,
        5.80583914e-01,  3.07634184e-01,  1.91067453e+00,  3.67363935e-01,
        3.34145454e-02,  4.37087871e-01])
>>> y_pred[:10,0,0]
array([0.28478737, 1.82613001, 0.65293924, 0.1912654 , 0.31614657,
       0.21313114, 2.05736439, 0.41425434, 0.20269812, 0.20916793])
>>> type(y_test)
<class 'numpy.ndarray'>
>>> type(y_pred)
<class 'numpy.ndarray'>
>>> y_test.dtype
dtype('float64')
>>> y_pred.dtype
dtype('float64')
>>> y_test[4:10,0]
array([0.58058391, 0.30763418, 1.91067453, 0.36736394, 0.03341455,
       0.43708787])
>>> y_pred[4:10,0]
array([[ 0.31614657,  0.07289431],
       [ 0.21313114,  0.04142069],
       [ 2.05736439, -0.22100096],
       [ 0.41425434,  0.00395289],
       [ 0.20269812,  0.02054913],
       [ 0.20916793,  0.03692715]])
>>> y_pred[4:10,0,0]
array([0.31614657, 0.21313114, 2.05736439, 0.41425434, 0.20269812,
       0.20916793])
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: 9512400852.6354 - val_loss: 11175283643.0121
Epoch 2/50
 - 3s - loss: 7966536314.8096 - val_loss: 10466896419.9209
Epoch 3/50
 - 2s - loss: 7292632101.6027 - val_loss: 7920053509.8589
Epoch 4/50
 - 2s - loss: 5502973992.6511 - val_loss: 6356192790.3077
Epoch 5/50
 - 2s - loss: 2715157525.9884 - val_loss: 4217800179.5122
Epoch 6/50
 - 2s - loss: 2055377223.9943 - val_loss: 2939734017.7813
Epoch 7/50
 - 2s - loss: 1513135286.1551 - val_loss: 2768994211.5508
Epoch 8/50
 - 2s - loss: 1068757593.2485 - val_loss: 1431877801.0939
Epoch 9/50
 - 2s - loss: 977787169.9193 - val_loss: 1167528781.1295
Epoch 10/50
 - 1s - loss: 820959968.5282 - val_loss: 1108924585.1475
Epoch 11/50
 - 1s - loss: 636864508.5249 - val_loss: 1075758026.0664
Epoch 12/50
 - 1s - loss: 612855521.4489 - val_loss: 1041913662.8195
Epoch 13/50
 - 1s - loss: 594986349.6722 - val_loss: 1008699886.8496
Epoch 14/50
 - 3s - loss: 582221138.4515 - val_loss: 981641216.0481
Epoch 15/50
 - 3s - loss: 278223082.8600 - val_loss: 814697562.3332
Epoch 16/50
 - 2s - loss: 419360721.7675 - val_loss: 826136.1333
Epoch 17/50
 - 2s - loss: 397659884.9647 - val_loss: 100241.7741
Epoch 18/50
 - 2s - loss: 41303918.0979 - val_loss: 248255.2257
Epoch 19/50
 - 2s - loss: 398447201.9312 - val_loss: 12595.6873
Epoch 20/50
 - 1s - loss: 22940438.9274 - val_loss: 32724.8753
Epoch 21/50
 - 1s - loss: 376436916.4687 - val_loss: 450977.1241
Epoch 22/50
 - 1s - loss: 94252344.5560 - val_loss: 3380.5397
Epoch 23/50
 - 3s - loss: 204067649.0782 - val_loss: 21500.3995
Epoch 24/50
 - 2s - loss: 9827047.1707 - val_loss: 44628.0976
Epoch 25/50
 - 2s - loss: 93079254.3101 - val_loss: 165.1674
Epoch 26/50
 - 1s - loss: 257099127.6840 - val_loss: 8303.6906
Epoch 27/50
 - 1s - loss: 9810046.3287 - val_loss: 3535.9004
Epoch 28/50
 - 1s - loss: 94636703.3885 - val_loss: 3497.3527
Epoch 29/50
 - 3s - loss: 9011930.6677 - val_loss: 4153.1335
Epoch 30/50
 - 2s - loss: 329719195.7545 - val_loss: 264.9710
Epoch 31/50
 - 2s - loss: 262911927.5658 - val_loss: 594.9686
Epoch 32/50
 - 2s - loss: 3300491.1672 - val_loss: 193.9337
Epoch 33/50
 - 2s - loss: 4496401.5001 - val_loss: 106.0950
Epoch 34/50
 - 2s - loss: 359663998.2473 - val_loss: 6571.7517
Epoch 35/50
 - 1s - loss: 13205279.3828 - val_loss: 359.4503
Epoch 36/50
 - 1s - loss: 3582198.5013 - val_loss: 112564.2320
Epoch 37/50
 - 2s - loss: 4855877.6513 - val_loss: 622.8318
Epoch 38/50
 - 2s - loss: 105477247.7464 - val_loss: 239.4771
Epoch 39/50
 - 1s - loss: 1669962.0461 - val_loss: 758.4415
Epoch 40/50
 - 1s - loss: 14213940.9527 - val_loss: 52188.5294
Epoch 41/50
 - 1s - loss: 377505849.4962 - val_loss: 56843.3652
Epoch 42/50
 - 1s - loss: 355134713.5367 - val_loss: 736271.2975
Epoch 43/50
 - 1s - loss: 77372788.2533 - val_loss: 123941.7445
Epoch 44/50
 - 1s - loss: 259643998.4612 - val_loss: 337.9006
Epoch 45/50
 - 1s - loss: 122615519.0755 - val_loss: 177.9063
Epoch 46/50
 - 3s - loss: 20017295.1686 - val_loss: 770.5466
Epoch 47/50
 - 3s - loss: 12421405.8333 - val_loss: 982.0977
Epoch 48/50
 - 2s - loss: 244003718.2578 - val_loss: 4031.8405
Epoch 49/50
 - 2s - loss: 35404896.0846 - val_loss: 238.7478
Epoch 50/50
 - 2s - loss: 2959270.6648 - val_loss: 3511.9257
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 2s - loss: 1557111463.1647 - val_loss: 5343239540.5718
Epoch 2/50
 - 1s - loss: 684187927.8848 - val_loss: 4919377770.2176
Epoch 3/50
 - 1s - loss: 973325828.2776 - val_loss: 169653354.0987
Epoch 4/50
 - 1s - loss: 1074667004.1861 - val_loss: 7988081.6182
Epoch 5/50
 - 1s - loss: 888986526.0445 - val_loss: 571610.8292
Epoch 6/50
 - 1s - loss: 641028736.4794 - val_loss: 52167.5517
Epoch 7/50
 - 1s - loss: 437811383.6572 - val_loss: 1043.3822
Epoch 8/50
 - 1s - loss: 614545912.3902 - val_loss: 259.3603
Epoch 9/50
 - 1s - loss: 576786755.6988 - val_loss: 293.4011
Epoch 10/50
 - 1s - loss: 858993553.3437 - val_loss: 229.6858
Epoch 11/50
 - 1s - loss: 552889314.9048 - val_loss: 230.1413
Epoch 12/50
 - 1s - loss: 58003987.9599 - val_loss: 226.7674
Epoch 13/50
 - 1s - loss: 5043053.3818 - val_loss: 232.1968
Epoch 14/50
 - 1s - loss: 77924363.5761 - val_loss: 226.7049
Epoch 15/50
 - 1s - loss: 2331600.2538 - val_loss: 225.1557
Epoch 16/50
 - 1s - loss: 13858325.5545 - val_loss: 224.6922
Epoch 17/50
 - 1s - loss: 444927.7285 - val_loss: 224.0979
Epoch 18/50
 - 1s - loss: 21333.2642 - val_loss: 223.6220
Epoch 19/50
 - 1s - loss: 3606.4938 - val_loss: 223.1924
Epoch 20/50
 - 1s - loss: 222.5502 - val_loss: 221.7252
Epoch 21/50
 - 1s - loss: 221.2201 - val_loss: 220.5139
Epoch 22/50
 - 1s - loss: 219.6261 - val_loss: 218.6906
Epoch 23/50
 - 1s - loss: 218.1243 - val_loss: 217.7484
Epoch 24/50
 - 1s - loss: 217.5053 - val_loss: 217.3677
Epoch 25/50
 - 1s - loss: 217.4552 - val_loss: 217.1988
Epoch 26/50
 - 1s - loss: 217.4467 - val_loss: 217.4957
Epoch 27/50
 - 1s - loss: 217.4747 - val_loss: 217.1627
Epoch 28/50
 - 1s - loss: 217.4433 - val_loss: 217.4930
Epoch 29/50
 - 1s - loss: 217.4472 - val_loss: 217.1170
Epoch 30/50
 - 1s - loss: 217.4397 - val_loss: 217.5241
Epoch 31/50
 - 1s - loss: 217.4426 - val_loss: 217.7466
Epoch 32/50
 - 1s - loss: 217.4361 - val_loss: 217.5893
Epoch 33/50
 - 1s - loss: 217.4371 - val_loss: 217.2841
Epoch 34/50
 - 1s - loss: 217.4555 - val_loss: 217.4506
Epoch 35/50
 - 1s - loss: 217.4548 - val_loss: 217.5413
Epoch 36/50
 - 1s - loss: 217.4505 - val_loss: 217.4090
Epoch 37/50
 - 1s - loss: 217.4659 - val_loss: 217.5545
Epoch 38/50
 - 1s - loss: 217.4485 - val_loss: 217.2876
Epoch 39/50
 - 1s - loss: 217.4703 - val_loss: 217.3708
Epoch 40/50
 - 1s - loss: 217.4596 - val_loss: 217.7558
Epoch 41/50
 - 1s - loss: 217.4849 - val_loss: 217.7806
Epoch 42/50
 - 1s - loss: 217.4232 - val_loss: 217.5449
Epoch 43/50
 - 1s - loss: 217.4836 - val_loss: 217.5851
Epoch 44/50
 - 1s - loss: 217.4533 - val_loss: 217.4253
Epoch 45/50
 - 1s - loss: 217.4737 - val_loss: 217.7506
Epoch 46/50
 - 1s - loss: 217.4560 - val_loss: 217.4542
Epoch 47/50
 - 1s - loss: 217.5004 - val_loss: 217.5113
Epoch 48/50
 - 1s - loss: 217.4451 - val_loss: 217.5144
Epoch 49/50
 - 1s - loss: 217.4615 - val_loss: 217.1242
Epoch 50/50
 - 1s - loss: 217.4558 - val_loss: 217.5286
>>> model.get_weights()
[array([[-0.00206519,  0.00118712, -0.00124588,  0.00312903,  0.00098851,
        -0.00163617, -0.00065436, -0.00016645],
       [-0.00283548,  0.00192833, -0.00164369,  0.00315525,  0.00267671,
        -0.00119966, -0.00185132, -0.00105243],
       [ 0.00114826, -0.00071502, -0.00258947,  0.00076766,  0.00082587,
         0.00292821, -0.00220332,  0.0005437 ],
       [ 0.00043032,  0.00189765, -0.00291432, -0.00047376,  0.002837  ,
         0.0031438 , -0.00031717, -0.00211243]], dtype=float32), array([[ 0.00255529,  0.0024887 ,  0.001761  , -0.00308266,  0.00290182,
         0.00229162, -0.0016825 ,  0.00143641],
       [-0.00150073, -0.00118587,  0.00217158,  0.0004602 , -0.00095398,
         0.00281495,  0.0020593 ,  0.00214105],
       [ 0.00090499,  0.00279853,  0.00146126, -0.00071744, -0.00140056,
        -0.00143652,  0.00076991,  0.00293338],
       [-0.00173978, -0.00067669, -0.0031358 , -0.00019071,  0.00061568,
         0.00258968, -0.00015892,  0.00065758],
       [ 0.00305464, -0.00131442, -0.00080785, -0.00067901, -0.00248346,
         0.00161491,  0.00222548, -0.00263017],
       [ 0.00090304,  0.00166866, -0.00044031,  0.00231078, -0.00107709,
        -0.00134383,  0.00043869,  0.00238095],
       [ 0.00141394,  0.00203718, -0.00208255, -0.00252994,  0.00260355,
         0.00217529,  0.00189839,  0.00116011],
       [ 0.00205807, -0.00270409, -0.00013113,  0.00267636,  0.00227974,
         0.00114112, -0.00191814,  0.00308129]], dtype=float32), array([[-1.2241013e-03,  7.5734453e-05, -2.8630900e-03,  3.7235493e-04,
         1.0981916e-03,  6.7331735e-04,  2.9369362e-03, -1.5370073e-03],
       [ 2.7603889e-03, -2.5559312e-03,  1.1029723e-03, -2.1188152e-03,
        -2.8571640e-03, -1.8672053e-03, -1.3732390e-03, -8.6847309e-04],
       [ 1.5364081e-03,  2.4609112e-03, -1.4849799e-03, -1.7944857e-03,
        -8.9380884e-04,  3.1350236e-03,  7.6569157e-04, -2.1467488e-03],
       [-1.6487631e-03,  2.0412463e-03, -2.2368655e-03, -1.2031408e-03,
        -1.3426237e-03, -1.3420784e-03, -2.8932253e-03, -7.3069101e-04],
       [ 1.7092330e-04, -1.9002242e-03, -2.2311567e-04, -9.3277567e-04,
         1.8187398e-03, -3.1085752e-03,  1.2873511e-03,  1.1745241e-03],
       [-3.1568815e-03,  2.4227174e-03,  3.1595547e-03, -1.9630697e-03,
        -2.8095099e-03,  1.2800957e-03, -2.2797862e-03,  2.2641849e-05],
       [-2.1119448e-03, -9.8967785e-04, -1.5319909e-03, -1.3590655e-03,
         2.7653340e-03, -2.8557517e-03,  1.5161443e-03,  3.0516747e-03],
       [ 2.7750118e-03,  1.7510218e-03, -2.7499569e-03,  3.1083836e-03,
         2.3579067e-03,  2.3935211e-03, -5.0218991e-04,  8.1408105e-04],
       [ 1.7342991e-03, -2.0816987e-03, -2.6190311e-03,  1.3350883e-03,
        -2.2337530e-03, -2.4814541e-03, -1.8436454e-03,  1.4088919e-03],
       [-3.1558671e-03, -4.9798569e-04,  2.9705036e-03, -5.3921330e-04,
        -8.8028272e-04, -1.7316501e-03, -2.5336598e-03, -2.0466303e-03],
       [-2.5677467e-03,  2.6188793e-03,  1.7707978e-03,  3.1337196e-03,
         1.7211661e-03, -2.0439510e-04, -4.1672905e-04,  8.8478380e-04],
       [-3.0326878e-03,  9.6408953e-04,  3.0357850e-04,  1.0835783e-03,
        -1.5444885e-04, -2.3302766e-03,  5.2036840e-04,  2.4295896e-03],
       [-2.9373730e-03, -2.8572155e-03,  1.0142261e-03,  1.2650524e-03,
        -2.5568977e-03, -3.1177884e-03,  3.0644285e-03,  1.6147163e-03],
       [-9.8258292e-04, -2.3102071e-04,  2.2397174e-03, -1.2906707e-03,
         3.8682332e-04, -3.1074537e-03, -3.1150605e-03, -8.7360450e-04],
       [ 1.1889002e-04,  2.8856527e-03,  2.5822180e-03,  9.6565823e-04,
        -5.0093187e-04, -1.4551213e-03,  2.4505407e-03,  2.6630387e-03],
       [-2.4116472e-03,  3.1047044e-04,  1.0196730e-03,  3.0123978e-03,
         1.2927968e-03,  2.7267059e-04, -9.1013528e-04, -2.3277213e-03]],
      dtype=float32), array([[ 5.3697652e-01],
       [-1.2375850e-03],
       [ 3.0570775e-03],
       [ 3.0909141e-04],
       [ 2.0186212e-03],
       [ 8.4761094e-04],
       [ 5.2181934e-04],
       [ 1.5984160e-03]], dtype=float32)]
>>> 8.7360450e-04
0.0008736045
>>> 0.0008736045**.01
0.9319940690296561
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
 - 2s - loss: nan - val_loss: nan
Epoch 4/50
 - 2s - loss: nan - val_loss: nan
Epoch 5/50
 - 2s - loss: nan - val_loss: nan
Epoch 6/50
 - 2s - loss: nan - val_loss: nan
Epoch 7/50
 - 2s - loss: nan - val_loss: nan
Epoch 8/50
 - 2s - loss: nan - val_loss: nan
Epoch 9/50
 - 1s - loss: nan - val_loss: nan
Epoch 10/50
 - 1s - loss: nan - val_loss: nan
Epoch 11/50
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 156, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.get_weights()
[array([[ 0.5130896 , -0.37512952,  0.43664992, -0.4710749 , -0.04730236,
         0.8016053 ,  0.0699141 ,  0.6831446 ],
       [-0.01914913,  0.40868753, -0.5053366 ,  0.21121812, -0.14024808,
         0.4465185 ,  0.01474571,  0.5037808 ],
       [ 0.6482154 ,  0.12346739, -0.04602991, -0.30803305, -0.8318351 ,
         0.16871409,  0.7461652 ,  0.15131298],
       [ 0.02099005,  0.03082413,  0.07882039, -0.6602142 , -0.23596755,
         0.6856481 , -0.6797217 , -0.61301476]], dtype=float32), array([[ 0.04697193,  0.13731425, -0.42216644,  0.1914246 , -0.45979884,
         0.113692  ,  0.20036033,  0.08043646],
       [-0.07014842,  0.184049  ,  0.25609902,  0.0025189 ,  0.12136371,
        -0.58002365, -0.31333834, -0.34861398],
       [-0.4608323 ,  0.06828501,  0.5581415 , -0.4503189 , -0.2980192 ,
        -0.07228754,  0.07245949, -0.30362016],
       [ 0.63680065,  0.02936986,  0.35950822,  0.09869545, -0.40641803,
         0.3200306 , -0.12421241,  0.22396278],
       [ 0.03197066,  0.14425333, -0.58646405,  0.03558038,  0.4307899 ,
        -0.00211413,  0.15189391,  0.1650615 ],
       [-0.16235076,  0.34549132,  0.4549312 , -0.0845383 , -0.02589523,
         0.20671794,  0.03642449, -0.08094596],
       [-0.38244495,  0.62324256,  0.18416014, -0.22318459, -0.42596114,
         0.19426617, -0.69918686,  0.27890438],
       [-0.5853911 , -0.6060526 , -0.63182765,  0.24942474, -0.22720514,
        -0.16985591,  0.75436884,  0.6665493 ]], dtype=float32), array([[-0.44933176,  0.06051448, -0.36546826, -0.5385063 ,  0.4154255 ,
         0.5506302 , -0.01561655,  0.00664587],
       [-0.4865899 ,  0.12768687, -0.05325323, -0.04366123, -0.30996084,
         0.04996713, -0.0956303 ,  0.31034404],
       [-0.1848399 , -0.23343247,  0.05612892, -0.27796572,  0.03816029,
         0.09800553, -0.05166659, -0.35124353],
       [-0.58132434, -0.31116173, -0.12315214, -0.39573652,  0.0834177 ,
        -0.5772151 ,  0.04972789, -0.10255419],
       [ 0.23013957, -0.37620112,  0.02386814,  0.23450892,  0.22016634,
        -0.23600721,  0.2686956 , -0.45527387],
       [-0.37329426, -0.43577123,  0.37628508,  0.5375341 , -0.12680209,
         0.17927225, -0.2130783 ,  0.16888234],
       [ 0.43360862,  0.5805137 , -0.23067948, -0.2228594 , -0.12488464,
        -0.13376573, -0.23207572, -0.2740969 ],
       [ 0.40400532,  0.08633576,  0.12748706,  0.05753422,  0.06260787,
         0.381345  , -0.4817777 , -0.05954283],
       [ 0.04988163,  0.28949553,  0.35538203,  0.02940237, -0.42563108,
         0.20050359, -0.34746265,  0.1142307 ],
       [-0.23975988,  0.4197906 ,  0.05607042,  0.42483163, -0.03027854,
        -0.46685964, -0.15436281,  0.20173025],
       [-0.05754266, -0.06481461, -0.06829036,  0.3642893 , -0.55263245,
         0.04754488,  0.23934172, -0.1293956 ],
       [ 0.00238985,  0.08120728,  0.08990264,  0.25325873, -0.22151451,
         0.19321455,  0.27616686,  0.10075278],
       [ 0.1558071 , -0.04030222, -0.04683695, -0.14712276,  0.4073863 ,
        -0.6316893 , -0.02555828,  0.4533647 ],
       [-0.2301484 , -0.41419128, -0.00528291,  0.3290824 , -0.05282065,
        -0.16720758,  0.05048309, -0.63826793],
       [-0.4997386 ,  0.49068624,  0.05767764, -0.3370671 ,  0.20260109,
         0.5716963 ,  0.25155517, -0.20296301],
       [ 0.07876095,  0.06357861,  0.12322248, -0.47144336,  0.01760606,
         0.2556938 ,  0.18236859,  0.25789085]], dtype=float32), array([[ 0.05199149],
       [-0.5727006 ],
       [ 0.20341437],
       [-0.39911202],
       [-0.6388489 ],
       [ 0.46120676],
       [ 1.0532639 ],
       [ 0.36657712]], dtype=float32)]
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.get_weights()
[array([[-0.6500732 , -0.55619407, -0.02089137, -0.44041803, -0.39433703,
         0.3781739 , -0.20440918,  0.34921223],
       [ 0.10048687, -0.6170364 ,  0.53012294, -0.6257062 , -0.49680865,
         0.06580675,  0.27581537, -0.59302264],
       [ 0.6635855 , -0.2968035 , -0.15044451,  0.05137986,  0.23904258,
        -0.40564537, -0.3978391 ,  0.34006268],
       [ 0.65427667,  0.5960191 , -0.6361743 , -0.01304901, -0.70018876,
        -0.35150826,  0.61171633,  0.65224737]], dtype=float32), array([[-0.05078393, -0.44071823, -0.5769316 ,  0.2870131 , -0.43778756,
         0.2253353 , -0.5568312 ,  0.41257042],
       [-0.47746187,  0.47631925,  0.20445418,  0.40784365, -0.46801573,
         0.5023001 , -0.07859904,  0.5260622 ],
       [ 0.08850169,  0.14490306,  0.48608333, -0.09877139,  0.15966624,
         0.5670987 , -0.15240473, -0.29026702],
       [ 0.5494024 , -0.5075899 ,  0.48601705,  0.51169735,  0.08394021,
         0.065772  ,  0.02334762, -0.23752007],
       [-0.6095616 ,  0.1952833 , -0.31414342, -0.37609223, -0.46837986,
        -0.3478574 ,  0.40472168, -0.30491528],
       [ 0.5044469 , -0.4881398 , -0.30539796, -0.57754743,  0.46404582,
         0.11274546,  0.5186742 , -0.30573493],
       [-0.47194242,  0.13661003,  0.14844388,  0.10851783, -0.06501961,
        -0.54997003,  0.02557439, -0.30300486],
       [-0.6028987 , -0.32436073, -0.04729587,  0.4541424 ,  0.14273494,
        -0.05722946,  0.0823164 , -0.45555925]], dtype=float32), array([[-0.13688052,  0.22518182, -0.20338893, -0.36056757,  0.04554713,
        -0.43188143,  0.45344865, -0.0278002 ],
       [ 0.02336645, -0.22609746,  0.41420007,  0.2371329 ,  0.4648955 ,
        -0.30671942,  0.4367453 , -0.2577225 ],
       [ 0.0421685 , -0.12994397,  0.17077744,  0.02690887, -0.12083948,
         0.29865754,  0.16770649, -0.11483741],
       [-0.45507312, -0.08002269, -0.23590457, -0.25774455, -0.4413936 ,
         0.14450109,  0.49177718, -0.49029374],
       [ 0.3948226 , -0.43472266, -0.36488628, -0.1335733 , -0.06884778,
         0.01077282,  0.26607072, -0.1110853 ],
       [ 0.1805954 , -0.42923653,  0.01762617, -0.1730547 , -0.3684839 ,
        -0.12941456, -0.485888  ,  0.46082044],
       [-0.08044755, -0.38379133, -0.17055404,  0.29983032,  0.08530438,
        -0.39021564,  0.4170165 ,  0.35700142],
       [-0.23366165,  0.21737361, -0.3771441 , -0.143574  ,  0.49752927,
        -0.07542324, -0.4183972 , -0.20849001],
       [-0.3860321 , -0.15772605,  0.19439852, -0.0703938 ,  0.4162401 ,
        -0.1763593 ,  0.30688238,  0.36071384],
       [-0.10509646,  0.35915458, -0.2701435 ,  0.24606013, -0.03594375,
         0.39126444, -0.49353385, -0.07189202],
       [ 0.08633459,  0.49280775,  0.3969592 ,  0.46673644, -0.18016791,
         0.40869784,  0.49248433, -0.24061239],
       [-0.30270147, -0.04039526, -0.30399358,  0.21780288,  0.37068605,
        -0.4689231 ,  0.1612364 , -0.1054064 ],
       [-0.25820363,  0.4593128 , -0.47632396, -0.25897586,  0.32017612,
         0.39279497,  0.48590052,  0.31046116],
       [-0.4005034 ,  0.47270834,  0.39951718,  0.36124635, -0.4562806 ,
         0.39369464,  0.3726369 ,  0.46277153],
       [ 0.20585752,  0.4966458 , -0.39679003,  0.14319038, -0.1747148 ,
         0.24997091, -0.17092681, -0.3080021 ],
       [ 0.23362768,  0.32350922, -0.30145764, -0.34946322,  0.14267123,
         0.43398142,  0.2996546 , -0.00997138]], dtype=float32), array([[-0.7599937 ],
       [-0.5137743 ],
       [-0.08406037],
       [ 0.7597679 ],
       [ 0.44785762],
       [ 0.27302313],
       [ 0.6722243 ],
       [ 0.51591337]], dtype=float32)]
>>> model.fit(x_train, y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 4s - loss: nan - val_loss: nan
<keras.callbacks.callbacks.History object at 0x000001538428E7B8>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 4s - loss: 254453572.7475 - val_loss: 842.2393
<keras.callbacks.callbacks.History object at 0x000001CDFAB0F8D0>
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 2s - loss: 31483.8667 - val_loss: 240.5963
Epoch 2/50
 - 2s - loss: 4930.5972 - val_loss: 228.1432
Epoch 3/50
 - 2s - loss: 405.2219 - val_loss: 225.4920
Epoch 4/50
 - 1s - loss: 231.3182 - val_loss: 223.9814
Epoch 5/50
 - 1s - loss: 223.7673 - val_loss: 222.3215
Epoch 6/50
 - 1s - loss: 221.6665 - val_loss: 220.8190
Epoch 7/50
 - 1s - loss: 220.1050 - val_loss: 219.6799
Epoch 8/50
 - 1s - loss: 219.0278 - val_loss: 218.4258
Epoch 9/50
 - 1s - loss: 218.2388 - val_loss: 218.0587
Epoch 10/50
 - 1s - loss: 217.9355 - val_loss: 217.6627
Epoch 11/50
 - 2s - loss: 217.8040 - val_loss: 217.6470
Epoch 12/50
 - 2s - loss: 217.7301 - val_loss: 217.6150
Epoch 13/50
 - 2s - loss: 217.6881 - val_loss: 217.5571
Epoch 14/50
 - 2s - loss: 217.6667 - val_loss: 217.4799
Epoch 15/50
 - 2s - loss: 217.6267 - val_loss: 217.4602
Epoch 16/50
 - 2s - loss: 217.6685 - val_loss: 217.5751
Epoch 17/50
 - 2s - loss: 217.5979 - val_loss: 217.6684
Epoch 18/50
 - 2s - loss: 217.5241 - val_loss: 217.3982
Epoch 19/50
 - 2s - loss: 217.4617 - val_loss: 217.5317
Epoch 20/50
 - 2s - loss: 217.4630 - val_loss: 217.7385
Epoch 21/50
 - 1s - loss: 217.4807 - val_loss: 217.4965
Epoch 22/50
 - 2s - loss: 217.4628 - val_loss: 217.3992
Epoch 23/50
 - 1s - loss: 217.4941 - val_loss: 217.5668
Epoch 24/50
 - 1s - loss: 217.4520 - val_loss: 217.5353
Epoch 25/50
Traceback (most recent call last):
  File "<pyshell#660>", line 1, in <module>
    model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.fit(x_train, y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 4s - loss: 10929249223.3872 - val_loss: 7229292311.8728
Epoch 2/50
 - 2s - loss: 7952591288.9271 - val_loss: 3593155737.7535
Epoch 3/50
 - 2s - loss: 6973439564.0918 - val_loss: 3192965541.6617
Epoch 4/50
 - 2s - loss: 3997638378.2119 - val_loss: 294536718.1883
Epoch 5/50
 - 2s - loss: 4745376630.8918 - val_loss: 125650133.3069
Epoch 6/50
 - 2s - loss: 3218077238.9317 - val_loss: 63257511.1061
Epoch 7/50
 - 2s - loss: 3777354415.2757 - val_loss: 7752543.3785
Epoch 8/50
 - 2s - loss: 3048803697.0199 - val_loss: 27645230.1725
Epoch 9/50
 - 2s - loss: 2450180727.6542 - val_loss: 14076023.8407
Epoch 10/50
 - 1s - loss: 3853584010.9966 - val_loss: 52209273.8954
Epoch 11/50
 - 1s - loss: 3743975375.3973 - val_loss: 164562.5037
Epoch 12/50
 - 1s - loss: 3570317801.0380 - val_loss: 63844.9619
Epoch 13/50
 - 1s - loss: 3765092940.1080 - val_loss: 28951.1992
Epoch 14/50
 - 2s - loss: 2582625046.7703 - val_loss: 40002.5854
Epoch 15/50
 - 2s - loss: 2672825758.0423 - val_loss: 32939.0680
Epoch 16/50
 - 2s - loss: 1953728604.7423 - val_loss: 22715.2871
Epoch 17/50
 - 2s - loss: 2504358382.7001 - val_loss: 22778.5638
Epoch 18/50
 - 2s - loss: 1808354208.8943 - val_loss: 22605.4566
Epoch 19/50
 - 2s - loss: 1072494454.8555 - val_loss: 22613.8061
Epoch 20/50
 - 2s - loss: 1142599096.9946 - val_loss: 22603.4213
Epoch 21/50
 - 2s - loss: 920265925.5098 - val_loss: 22551.6879
Epoch 22/50
 - 2s - loss: 1340784820.4103 - val_loss: 22523.6800
Epoch 23/50
 - 2s - loss: 1083378925.8808 - val_loss: 22499.9119
Epoch 24/50
 - 2s - loss: 879669785.9084 - val_loss: 22470.9335
Epoch 25/50
 - 2s - loss: 859204588.7067 - val_loss: 22477.4181
Epoch 26/50
 - 2s - loss: 380466326.4628 - val_loss: 22439.0635
Epoch 27/50
 - 2s - loss: 841939840.9962 - val_loss: 22390.0428
Epoch 28/50
 - 2s - loss: 321681028.5003 - val_loss: 22357.1558
Epoch 29/50
 - 2s - loss: 211212439.0147 - val_loss: 22315.5017
Epoch 30/50
 - 2s - loss: 311498896.8035 - val_loss: 22314.4917
Epoch 31/50
 - 2s - loss: 315184076.3702 - val_loss: 22304.1204
Epoch 32/50
 - 2s - loss: 304827518.1006 - val_loss: 22271.6922
Epoch 33/50
 - 2s - loss: 301333358.4664 - val_loss: 22226.1124
Epoch 34/50
 - 2s - loss: 294072943.6825 - val_loss: 22201.0619
Epoch 35/50
 - 2s - loss: 7049919.7972 - val_loss: 22179.3570
Epoch 36/50
 - 2s - loss: 253446345.3652 - val_loss: 22184.2550
Epoch 37/50
 - 2s - loss: 285793577.3866 - val_loss: 22162.3171
Epoch 38/50
 - 2s - loss: 277574150.8653 - val_loss: 22171.4667
Epoch 39/50
 - 2s - loss: 24674241.2440 - val_loss: 22157.1046
Epoch 40/50
 - 1s - loss: 283952934.9527 - val_loss: 22074.5589
Epoch 41/50
 - 2s - loss: 61976270.5702 - val_loss: 22073.7000
Epoch 42/50
 - 2s - loss: 22118.5613 - val_loss: 22034.1278
Epoch 43/50
 - 2s - loss: 21968.1167 - val_loss: 21888.8200
Epoch 44/50
 - 2s - loss: 21835.2544 - val_loss: 21784.6302
Epoch 45/50
 - 1s - loss: 21750.1382 - val_loss: 21719.1028
Epoch 46/50
 - 1s - loss: 21717.8082 - val_loss: 21682.8195
Epoch 47/50
 - 2s - loss: 21712.7508 - val_loss: 21723.7279
Epoch 48/50
 - 2s - loss: 21711.4866 - val_loss: 21726.4961
Epoch 49/50
 - 2s - loss: 21715.3696 - val_loss: 21723.3979
Epoch 50/50
 - 2s - loss: 21711.1938 - val_loss: 21726.2042
<keras.callbacks.callbacks.History object at 0x000001A3A6AE5DA0>
>>> model.get_weights[0]
Traceback (most recent call last):
  File "<pyshell#662>", line 1, in <module>
    model.get_weights[0]
TypeError: 'method' object is not subscriptable
>>> model.get_weights()[0]
array([[ 0.00093552,  0.00098235,  0.00097778,  0.00062466,  0.00167313,
         0.00282131,  0.0013117 ,  0.0010384 ],
       [ 0.00191629, -0.00222531,  0.00132823, -0.00150624, -0.00113884,
         0.002168  , -0.0025269 , -0.0003858 ],
       [-0.00210027, -0.00106664,  0.00255025, -0.0029808 , -0.00283459,
         0.00161555, -0.00036552, -0.00164764],
       [-0.00211497, -0.00223846, -0.00044734,  0.00214584, -0.00169716,
         0.0026007 ,  0.00172637,  0.0026417 ]], dtype=float32)
>>> model.get_weights()[1]
array([[ 2.2347574e-04,  1.4578336e-03,  8.3158771e-04, -6.4560107e-04,
         1.1639635e-03,  1.6865187e-03, -2.2016433e-03,  1.6934829e-03],
       [-2.1077094e-03, -2.5271650e-03,  2.2023374e-03, -6.1787840e-05,
        -2.5713537e-03,  2.6772714e-03, -1.7593851e-03, -2.3443638e-03],
       [-7.2788761e-04, -1.5033530e-03,  3.0961307e-03, -1.1720156e-03,
        -1.2149375e-03, -1.0956465e-03,  1.5516849e-03, -1.7085123e-03],
       [-2.9267876e-03, -9.2912407e-04,  2.1759793e-03, -1.6011538e-03,
         1.5588466e-03,  1.7061591e-03, -2.3441175e-03, -9.6403703e-05],
       [ 1.2799501e-03,  3.1588413e-03,  1.4619920e-03,  2.9417467e-03,
        -2.8545216e-03,  9.2453032e-04, -1.5788798e-03,  3.1026544e-03],
       [ 1.8927943e-03, -2.3595223e-03, -2.2671639e-03,  6.4151845e-04,
        -6.2950095e-04, -1.7818180e-03, -3.1506247e-03, -1.8608540e-03],
       [-2.3698288e-03, -3.0663467e-03,  2.9637844e-03,  3.1224382e-03,
        -1.7567159e-03, -4.6904158e-04,  3.0602959e-03, -1.1382125e-03],
       [ 1.5501634e-03, -1.9717885e-03, -8.5841288e-04, -2.2753309e-03,
         5.4780918e-04,  8.7485102e-04, -2.9320538e-03,  1.5786443e-03]],
      dtype=float32)
>>> model.get_weights()[4]
Traceback (most recent call last):
  File "<pyshell#665>", line 1, in <module>
    model.get_weights()[4]
IndexError: list index out of range
>>> model.get_weights()[3]
array([[-0.00310869],
       [ 0.00094481],
       [-0.00245194],
       [ 0.00310492],
       [-0.00275483],
       [ 0.00215814],
       [-0.00281332],
       [ 0.00156366]], dtype=float32)
>>> weights = [w*0.0 for w in model.get_weights]
Traceback (most recent call last):
  File "<pyshell#667>", line 1, in <module>
    weights = [w*0.0 for w in model.get_weights]
TypeError: 'method' object is not iterable
>>> weights = [w*0.0 for w in model.get_weights()]
>>> weights
[array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0., -0.,  0., -0., -0.,  0., -0., -0.],
       [-0., -0.,  0., -0., -0.,  0., -0., -0.],
       [-0., -0., -0.,  0., -0.,  0.,  0.,  0.]], dtype=float32), array([[ 0.,  0.,  0., -0.,  0.,  0., -0.,  0.],
       [-0., -0.,  0., -0., -0.,  0., -0., -0.],
       [-0., -0.,  0., -0., -0., -0.,  0., -0.],
       [-0., -0.,  0., -0.,  0.,  0., -0., -0.],
       [ 0.,  0.,  0.,  0., -0.,  0., -0.,  0.],
       [ 0., -0., -0.,  0., -0., -0., -0., -0.],
       [-0., -0.,  0.,  0., -0., -0.,  0., -0.],
       [ 0., -0., -0., -0.,  0.,  0., -0.,  0.]], dtype=float32), array([[-0.,  0.,  0.,  0.,  0.,  0., -0.,  0.],
       [-0.,  0.,  0., -0., -0.,  0.,  0., -0.],
       [-0., -0., -0., -0., -0.,  0.,  0.,  0.],
       [-0.,  0.,  0., -0.,  0., -0., -0.,  0.],
       [-0., -0., -0., -0.,  0.,  0., -0.,  0.],
       [ 0., -0.,  0., -0., -0., -0., -0.,  0.],
       [-0., -0., -0.,  0., -0.,  0., -0., -0.],
       [-0.,  0.,  0.,  0.,  0.,  0., -0., -0.],
       [-0., -0.,  0.,  0., -0.,  0.,  0., -0.],
       [-0.,  0.,  0.,  0., -0., -0., -0.,  0.],
       [-0., -0.,  0.,  0.,  0., -0.,  0.,  0.],
       [ 0.,  0., -0.,  0., -0., -0.,  0.,  0.],
       [-0.,  0., -0.,  0., -0.,  0.,  0.,  0.],
       [-0.,  0., -0., -0.,  0.,  0., -0.,  0.],
       [-0., -0.,  0., -0.,  0., -0.,  0.,  0.],
       [-0.,  0., -0., -0., -0., -0.,  0., -0.]], dtype=float32), array([[-0.],
       [ 0.],
       [-0.],
       [ 0.],
       [-0.],
       [ 0.],
       [-0.],
       [ 0.]], dtype=float32)]
>>> weights = tf.abs(weights)
Traceback (most recent call last):
  File "<pyshell#670>", line 1, in <module>
    weights = tf.abs(weights)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
w  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 266, in abs
    x = ops.convert_to_tensor(x, name="x")
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1100, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
e  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 254, in _constant_impl
    t = convert_to_eager_tensor(value, ctx, dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 115, in convert_to_eager_tensor
    return ops.EagerTensor(value, handle, device, dtype)
ValueError: Can't convert non-rectangular Python sequence to Tensor.
>>> weights = [tf.zeros_like(w) for w in model.get_weights()]
>>> weights
[<tf.Tensor: id=80859, shape=(4, 8), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>, <tf.Tensor: id=80861, shape=(8, 8), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>, <tf.Tensor: id=80863, shape=(16, 8), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>, <tf.Tensor: id=80865, shape=(8, 1), dtype=float32, numpy=
array([[0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.]], dtype=float32)>]
>>> weights[0,0,0]
Traceback (most recent call last):
  File "<pyshell#673>", line 1, in <module>
    weights[0,0,0]
TypeError: list indices must be integers or slices, not tuple
>>> weights[0]
<tf.Tensor: id=80859, shape=(4, 8), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>
>>> weights[0].shape
TensorShape([4, 8])
>>> weights[0][3,7]
<tf.Tensor: id=80874, shape=(), dtype=float32, numpy=0.0>
>>> weights[0]=[[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]]
>>> weights[0]
[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
>>> weights
[[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], <tf.Tensor: id=80861, shape=(8, 8), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>, <tf.Tensor: id=80863, shape=(16, 8), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>, <tf.Tensor: id=80865, shape=(8, 1), dtype=float32, numpy=
array([[0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.]], dtype=float32)>]
>>> weights[0]=np.array(
      [[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]])
>>> weights
[array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]]), <tf.Tensor: id=80861, shape=(8, 8), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>, <tf.Tensor: id=80863, shape=(16, 8), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>, <tf.Tensor: id=80865, shape=(8, 1), dtype=float32, numpy=
array([[0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.]], dtype=float32)>]
>>> weights[0]=np.array(
      [[1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 2., 0., 0., 0., 0., 0.],
       [0., 1., 0., 2., 0., 0., 0., 0.]])
>>> weights[1]=np.array(
      [[1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]])
>>> weights[2]=np.array(
      [[0., 0., 0., 0., 0., 0., 0., 0.],
       [0.09531, 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0.5, 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]])
>>> weights[3]
<tf.Tensor: id=80865, shape=(8, 1), dtype=float32, numpy=
array([[0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.]], dtype=float32)>
>>> weights[3]=np.array([
       [1.],
       [1.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.]])
>>> weights
[array([[1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 2., 0., 0., 0., 0., 0.],
       [0., 1., 0., 2., 0., 0., 0., 0.]]), array([[1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]]), array([[0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.09531, 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [1.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.5    , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ]]), array([[1.],
       [1.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.]])]
>>> len(weights)
4
>>> for k in len(weights):
	for r in weights[k].shape[0]:
		for c in weights[k].shape[1]:
			if weights[k][r,c]!=0:
				print('weights[',k,'][',r,',',c,'] = ',weights[k][r,c])

				
Traceback (most recent call last):
  File "<pyshell#694>", line 1, in <module>
    for k in len(weights):
TypeError: 'int' object is not iterable
>>> for k in range(len(weights)):
	for r in range(weights[k].shape[0]):
		for c in wrange(eights[k].shape[1]):
			if weights[k][r,c]!=0:
				print('weights[',k,'][',r,',',c,'] = ',weights[k][r,c])

				
Traceback (most recent call last):
  File "<pyshell#696>", line 3, in <module>
    for c in wrange(eights[k].shape[1]):
NameError: name 'wrange' is not defined
>>> for k in range(len(weights)):
	for r in range(weights[k].shape[0]):
		for c in range(weights[k].shape[1]):
			if weights[k][r,c]!=0:
				print('weights[',k,'][',r,',',c,'] = ',weights[k][r,c])

				
weights[ 0 ][ 0 , 0 ] =  1.0
weights[ 0 ][ 2 , 1 ] =  1.0
weights[ 0 ][ 2 , 2 ] =  2.0
weights[ 0 ][ 3 , 1 ] =  1.0
weights[ 0 ][ 3 , 3 ] =  2.0
weights[ 1 ][ 0 , 0 ] =  1.0
weights[ 1 ][ 1 , 1 ] =  1.0
weights[ 1 ][ 2 , 2 ] =  1.0
weights[ 1 ][ 3 , 2 ] =  1.0
weights[ 2 ][ 1 , 0 ] =  0.09531
weights[ 2 ][ 3 , 0 ] =  1.0
weights[ 2 ][ 5 , 1 ] =  0.5
weights[ 3 ][ 0 , 0 ] =  1.0
weights[ 3 ][ 1 , 0 ] =  1.0
>>> indices = []
>>> values = []
>>> for k in range(len(weights)):
	for r in range(weights[k].shape[0]):
		for c in range(weights[k].shape[1]):
			if weights[k][r,c]!=0:
				indices.append((k,,r,c,weights[k][r,c]))
				
SyntaxError: invalid syntax
for k in range(len(weights)):
	for r in range(weights[k].shape[0]):
		for c in range(weights[k].shape[1]):
			if weights[k][r,c]!=0:
				indices.append((k,,r,c,weights[k][r,c]))
>>> for k in range(len(weights)):
	for r in range(weights[k].shape[0]):
		for c in range(weights[k].shape[1]):
			if weights[k][r,c]!=0:
				indices.append((k,r,c,weights[k][r,c]))

				
>>> indices
[(0, 0, 0, 1.0), (0, 2, 1, 1.0), (0, 2, 2, 2.0), (0, 3, 1, 1.0), (0, 3, 3, 2.0), (1, 0, 0, 1.0), (1, 1, 1, 1.0), (1, 2, 2, 1.0), (1, 3, 2, 1.0), (2, 1, 0, 0.09531), (2, 3, 0, 1.0), (2, 5, 1, 0.5), (3, 0, 0, 1.0), (3, 1, 0, 1.0)]
>>> tf.zeros(4,8)
<tf.Tensor: id=80885, shape=(4,), dtype=complex64, numpy=array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex64)>
>>> tf.zeros(4,8,float32)
Traceback (most recent call last):
  File "<pyshell#706>", line 1, in <module>
    tf.zeros(4,8,float32)
NameError: name 'float32' is not defined
>>> tf.zeros(4,8,tf.float32)
Traceback (most recent call last):
  File "<pyshell#707>", line 1, in <module>
    tf.zeros(4,8,tf.float32)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\array_ops.py", line 1845, in zeros
    with ops.name_scope(name, "zeros", [shape]) as name:
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 6168, in __enter__
    elif self._name[-1] == "/":
TypeError: 'DType' object does not support indexing
>>> tf.zeros((4,8))
<tf.Tensor: id=80889, shape=(4, 8), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 168, in <module>
    model.set_weights(create_weights())
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 142, in create_weights
    weights[s[0]][s[1],s[2]]=s[3]
TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 168, in <module>
    model.set_weights(create_weights())
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\network.py", line 527, in set_weights
    K.batch_set_value(tuples)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 2960, in batch_set_value
    tf_keras_backend.batch_set_value(tuples)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3056, in batch_set_value
    x.assign(np.asarray(value, dtype=dtype(x)))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py", line 1145, in assign
    self._shape.assert_is_compatible_with(value_tensor.shape)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_shape.py", line 1110, in assert_is_compatible_with
    raise ValueError("Shapes %s and %s are incompatible" % (self, other))
ValueError: Shapes (16, 8) and (18, 8) are incompatible
>>> create_weights()
[array([[1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 2., 0., 0., 0., 0., 0.],
       [0., 1., 0., 2., 0., 0., 0., 0.]]), array([[1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]]), array([[0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.09531, 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [1.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.5    , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ]]), array([[1.],
       [1.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.]])]
>>> y_test.shape
(2000, 1)
>>> def create_weights():
    weights = [np.zeros((4,8)),np.zeros((8,8)),np.zeros((16,8)),np.zeros((8,1))]
    sparse = [(0,0,0,1.0),(0,2,1,1.0),(0,2,2,2.0),(0,3,1,1.0),(0,3,3,2.0),(1,0,0,1.0),
              (1,1,1,1.0),(1,2,2,1.0),(1,3,2,1.0),(2,1,0,0.09531),(2,3,0,1.0),(2,5,1,0.5),(3,0,0,1.0),(3,1,0,1.0)]
    for s in sparse:
        weights[s[0]][s[1],s[2]]=s[3]
    return weights

>>> model.set_weights(create_weights())
>>> y_pred=model.predict(x_test)
>>> complex_mean_absolute_error(y_test,y_pred)
<tf.Tensor: id=2838, shape=(), dtype=float64, numpy=0.6258230052022611>
>>> model = Model(inputs=[value_in], outputs=[dense_1])
>>> model.set_weights(create_weights()[:2])
>>> y_pred=model.predict(x_test)
>>> x_test[:10]
array([[ 0.37330565, -0.29237621],
       [ 0.18513376, -0.21819253],
       [ 0.646554  ,  0.95844926],
       [ 0.42402005,  0.74045555],
       [ 0.47585845, -0.47873455],
       [ 0.19481104, -0.25495621],
       [ 0.59049543, -0.21957586],
       [-0.41263376, -0.31838897],
       [ 0.05457677,  0.78262626],
       [-0.63735057, -0.4223744 ]])
>>> y_pred[:10]
array([[[ 1.45252824e+00,  0.00000000e+00],
        [-1.09145695e-01,  1.33664925e-17],
        [ 2.24840961e-01, -2.09375057e-17],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 1.20337941e+00,  0.00000000e+00],
        [-4.03948062e-02,  4.94693701e-18],
        [ 8.18824941e-02, -1.16605924e-17],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 1.90895122e+00,  0.00000000e+00],
        [ 6.19689186e-01,  0.00000000e+00],
        [ 1.33665702e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 1.52809223e+00,  0.00000000e+00],
        [ 3.13968008e-01,  0.00000000e+00],
        [ 7.28067452e-01,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 1.60939519e+00,  0.00000000e+00],
        [-2.27809882e-01,  2.78986643e-17],
        [ 4.55628036e-01, -5.61345694e-17],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 1.21508137e+00,  0.00000000e+00],
        [-4.96682871e-02,  6.08261088e-18],
        [ 1.02954016e-01, -1.59210628e-17],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 1.80488235e+00,  0.00000000e+00],
        [-1.29658532e-01,  1.58785907e-17],
        [ 3.96898381e-01, -1.18089151e-17],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 6.61904643e-01,  0.00000000e+00],
        [ 1.31378043e-01, -3.21783399e-17],
        [ 2.71638169e-01, -6.65321628e-17],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 1.05609355e+00,  0.00000000e+00],
        [ 4.27132107e-02,  0.00000000e+00],
        [ 6.15482504e-01,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 5.28691306e-01,  0.00000000e+00],
        [ 2.69200558e-01, -6.59351205e-17],
        [ 5.84615867e-01, -1.43189590e-16],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]]])
>>> y_pred.shape
(2000, 8, 2)
>>> y_pred[:10,:3,0]
array([[ 1.45252824, -0.10914569,  0.22484096],
       [ 1.20337941, -0.04039481,  0.08188249],
       [ 1.90895122,  0.61968919,  1.33665702],
       [ 1.52809223,  0.31396801,  0.72806745],
       [ 1.60939519, -0.22780988,  0.45562804],
       [ 1.21508137, -0.04966829,  0.10295402],
       [ 1.80488235, -0.12965853,  0.39689838],
       [ 0.66190464,  0.13137804,  0.27163817],
       [ 1.05609355,  0.04271321,  0.6154825 ],
       [ 0.52869131,  0.26920056,  0.58461587]])
>>> x_test[:10]
array([[ 0.37330565, -0.29237621],
       [ 0.18513376, -0.21819253],
       [ 0.646554  ,  0.95844926],
       [ 0.42402005,  0.74045555],
       [ 0.47585845, -0.47873455],
       [ 0.19481104, -0.25495621],
       [ 0.59049543, -0.21957586],
       [-0.41263376, -0.31838897],
       [ 0.05457677,  0.78262626],
       [-0.63735057, -0.4223744 ]])
>>> 0.37330565* -0.29237621
-0.1091456911185865
>>> def create_weights():
    weights = [np.zeros((4,8)),np.zeros((8,8)),np.zeros((16,8)),np.zeros((8,1))]
    sparse = [(0,2,0,1.0),(0,2,1,1.0),(0,2,2,2.0),(0,3,1,1.0),(0,3,3,2.0),(1,0,0,1.0),
              (1,1,1,1.0),(1,2,2,1.0),(1,3,2,1.0),(2,1,0,0.09531),(2,3,0,1.0),(2,5,1,0.5),(3,0,0,1.0),(3,1,0,1.0)]
    for s in sparse:
        weights[s[0]][s[1],s[2]]=s[3]
    return weights
model.set_weights(create_weights()[:2])
SyntaxError: invalid syntax
>>> def create_weights():
    weights = [np.zeros((4,8)),np.zeros((8,8)),np.zeros((16,8)),np.zeros((8,1))]
    sparse = [(0,2,0,1.0),(0,2,1,1.0),(0,2,2,2.0),(0,3,1,1.0),(0,3,3,2.0),(1,0,0,1.0),
              (1,1,1,1.0),(1,2,2,1.0),(1,3,2,1.0),(2,1,0,0.09531),(2,3,0,1.0),(2,5,1,0.5),(3,0,0,1.0),(3,1,0,1.0)]
    for s in sparse:
        weights[s[0]][s[1],s[2]]=s[3]
    return weights

>>> model.set_weights(create_weights()[:2])
>>> y_pred=model.predict(x_test)
>>> y_pred[:10,:3,0]
array([[ 0.37330565, -0.10914569,  0.22484096],
       [ 0.18513377, -0.04039481,  0.08188249],
       [ 0.64655399,  0.61968919,  1.33665702],
       [ 0.42402005,  0.31396801,  0.72806745],
       [ 0.47585845, -0.22780988,  0.45562804],
       [ 0.19481105, -0.04966829,  0.10295402],
       [ 0.59049541, -0.12965853,  0.39689838],
       [-0.41263378,  0.13137804,  0.27163817],
       [ 0.05457677,  0.04271321,  0.6154825 ],
       [-0.63735056,  0.26920056,  0.58461587]])
>>> x_test[:10]
array([[ 0.37330565, -0.29237621],
       [ 0.18513376, -0.21819253],
       [ 0.646554  ,  0.95844926],
       [ 0.42402005,  0.74045555],
       [ 0.47585845, -0.47873455],
       [ 0.19481104, -0.25495621],
       [ 0.59049543, -0.21957586],
       [-0.41263376, -0.31838897],
       [ 0.05457677,  0.78262626],
       [-0.63735057, -0.4223744 ]])
>>> 0.37330565**2+0.29237621**2
0.2248409564958866
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> y_pred=model.predict(x_test)
>>> complex_mean_absolute_error(y_test, y_pred)
<tf.Tensor: id=2835, shape=(), dtype=float64, numpy=0.6216049303214208>
>>> model = Model(inputs=[value_in], outputs=[dense_1])
>>> model.set_weights(create_weights()[:2])
>>> y_pred=model.predict(x_test)
>>> x_test[:10]
array([[ 0.52403529,  0.7065142 ],
       [-0.9066728 ,  0.49259629],
       [ 0.23898349,  0.80655389],
       [ 0.10246866,  0.75650119],
       [-0.95626782,  0.21349879],
       [-0.23144448,  0.69870831],
       [-0.32128623,  0.42227306],
       [ 0.93341284, -0.16302738],
       [ 0.22865936, -0.33426374],
       [-0.27589337, -0.68338025]])
>>> y_pred[:10,:3,0]
array([[ 0.52403527,  0.37023835,  0.77377526],
       [-0.90667278, -0.44662365,  1.06470664],
       [ 0.23898348,  0.19275306,  0.7076423 ],
       [ 0.10246866,  0.07751767,  0.58279389],
       [-0.95626783, -0.20416202,  0.9600299 ],
       [-0.23144448, -0.16171218,  0.54175983],
       [-0.32128623, -0.13567052,  0.28153939],
       [ 0.93341285, -0.15217185,  0.89783747],
       [ 0.22865936, -0.07643253,  0.16401735],
       [-0.27589336,  0.18854007,  0.54312571]])
>>> 0.52403529*  0.7065142
0.370238373686118
>>> 0.52403529**2 + 0.7065142**2
0.7737752999670241
>>> model = Model(inputs=[value_in], outputs=[power_dense_2])
Traceback (most recent call last):
  File "<pyshell#743>", line 1, in <module>
    model = Model(inputs=[value_in], outputs=[power_dense_2])
NameError: name 'power_dense_2' is not defined
>>> model = Model(inputs=[value_in], outputs=[p_dense_2])
>>> model.set_weights(create_weights()[:3])
>>> y_pred=model.predict(x_test)
>>> x_test[:10]
array([[ 0.52403529,  0.7065142 ],
       [-0.9066728 ,  0.49259629],
       [ 0.23898349,  0.80655389],
       [ 0.10246866,  0.75650119],
       [-0.95626782,  0.21349879],
       [-0.23144448,  0.69870831],
       [-0.32128623,  0.42227306],
       [ 0.93341284, -0.16302738],
       [ 0.22865936, -0.33426374],
       [-0.27589337, -0.68338025]])
>>> y_pred[:10,:2,0]
array([[1.03591741, 1.        ],
       [0.95832558, 1.        ],
       [1.01854109, 1.        ],
       [1.00741557, 1.        ],
       [0.98072942, 1.        ],
       [0.98470538, 1.        ],
       [0.98715249, 1.        ],
       [0.98560117, 1.        ],
       [0.99274168, 1.        ],
       [1.01813218, 1.        ]])
>>> 0.52403529*1.1**(0.52403529*0.7065142)
0.5428573158597042
>>> y_pred[:10,:,0]
array([[1.03591741, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ],
       [0.95832558, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ],
       [1.01854109, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ],
       [1.00741557, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ],
       [0.98072942, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ],
       [0.98470538, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ],
       [0.98715249, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ],
       [0.98560117, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ],
       [0.99274168, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ],
       [1.01813218, 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        ]])
>>> y_pred[:10,:,:]
array([[[ 1.03591741e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]],

       [[ 9.58325583e-01,  4.99578954e-18],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]],

       [[ 1.01854109e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]],

       [[ 1.00741557e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]],

       [[ 9.80729415e-01,  2.33707953e-18],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]],

       [[ 9.84705381e-01,  1.85865322e-18],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]],

       [[ 9.87152485e-01,  1.56321630e-18],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]],

       [[ 9.85601170e-01,  1.75059167e-18],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]],

       [[ 9.92741685e-01,  8.85653510e-19],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]],

       [[ 1.01813218e+00, -4.48112608e-18],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00]]])
>>> model.get_weights()[2]
array([[0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.09531, 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [1.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.5    , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ]], dtype=float32)
>>> def create_weights():
    weights = [np.zeros((4,8)),np.zeros((8,8)),np.zeros((16,8)),np.zeros((8,1))]
    sparse = [(0,2,0,1.0),(0,2,1,1.0),(0,2,2,2.0),(0,3,1,1.0),(0,3,3,2.0),(1,0,0,1.0),
              (1,1,1,1.0),(1,2,2,1.0),(1,3,2,1.0),(2,1,0,0.09531),(2,8,0,1.0),(2,10,1,0.5),(3,0,0,1.0),(3,1,0,1.0)]
    for s in sparse:
        weights[s[0]][s[1],s[2]]=s[3]
    return weights

>>> model.set_weights(create_weights()[:3])
>>> y_pred=model.predict(x_test)
>>> x_test[:10]
array([[ 0.52403529,  0.7065142 ],
       [-0.9066728 ,  0.49259629],
       [ 0.23898349,  0.80655389],
       [ 0.10246866,  0.75650119],
       [-0.95626782,  0.21349879],
       [-0.23144448,  0.69870831],
       [-0.32128623,  0.42227306],
       [ 0.93341284, -0.16302738],
       [ 0.22865936, -0.33426374],
       [-0.27589337, -0.68338025]])
>>> y_pred[:10,:2,0]
array([[ 0.54285726,  0.87964496],
       [-0.86888772,  1.03184623],
       [ 0.2434145 ,  0.84121478],
       [ 0.10322853,  0.76340938],
       [-0.93783999,  0.97981116],
       [-0.22790462,  0.73604336],
       [-0.3171585 ,  0.53060285],
       [ 0.9199728 ,  0.94754286],
       [ 0.22699968,  0.40499056],
       [-0.28089591,  0.73697063]])
>>> np.sqrt(0.52403529**2+0.7065142**2)
0.8796449851883565
>>> 0.52403529*1.1**(0.52403529*0.7065142)
0.5428573158597042
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> y_pred=model.predict(x_test)
>>> complex_mean_absolute_error(y_test, y_pred)
<tf.Tensor: id=2835, shape=(), dtype=float64, numpy=1.757188460277081e-08>
>>> [l0_1(w) for w in model.get_weights()]
[<tf.Tensor: id=2851, shape=(), dtype=float64, numpy=5.013911247253418>, <tf.Tensor: id=2858, shape=(), dtype=float64, numpy=4.0>, <tf.Tensor: id=2865, shape=(), dtype=float64, numpy=2.969860315322876>, <tf.Tensor: id=2872, shape=(), dtype=float64, numpy=2.0>]
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)/10 for w in model.get_weights()])
>>> model.get_weights
<bound method Network.get_weights of <keras.engine.training.Model object at 0x000001CFA8EA4F60>>
>>> model.get_weights()
[array([[-0.2619333 , -0.07104148,  0.11841224, -0.05960658, -0.09646287,
        -0.01227169, -0.114448  , -0.13983679],
       [-0.05085285, -0.17926407, -0.12646168, -0.02238642,  0.00338151,
        -0.18158986, -0.2113138 , -0.06792036],
       [ 1.0107672 ,  0.92334634,  2.2152026 ,  0.10884906, -0.05493819,
         0.14664686,  0.04485006, -0.29658502],
       [ 0.17185268,  0.7978849 ,  0.00654925,  1.7819837 , -0.00670258,
        -0.15719086, -0.03127784, -0.23371705]], dtype=float32), array([[ 1.0790955 , -0.06450568, -0.19530281,  0.07483867, -0.11790257,
        -0.01853265, -0.09581172,  0.03865829],
       [-0.17634715,  0.9145078 , -0.04339519,  0.018392  , -0.233546  ,
         0.06476402, -0.05073986, -0.16832875],
       [ 0.02587353, -0.09045402,  0.90273607,  0.04610477, -0.10789178,
         0.04297636, -0.0147423 , -0.0318945 ],
       [-0.06484032, -0.10623214,  1.0315406 , -0.2547802 , -0.05128995,
        -0.16759217,  0.00851538,  0.07242728],
       [ 0.10853008, -0.0564002 ,  0.09156822, -0.07301874, -0.03767002,
        -0.01299692,  0.04604122, -0.1528453 ],
       [-0.06277829, -0.07596534, -0.08400559, -0.12096808, -0.06742445,
        -0.13034506, -0.15126862, -0.06698896],
       [-0.12849659, -0.09614041, -0.11625587,  0.0214608 , -0.02249013,
         0.04426993, -0.0401139 , -0.30143064],
       [ 0.04050533, -0.09942352, -0.13660319,  0.00328173, -0.19129387,
        -0.16542552, -0.07990137,  0.0356501 ]], dtype=float32), array([[ 0.09100579, -0.14399388, -0.04098491,  0.08591623, -0.02721918,
         0.13411556, -0.24579452,  0.0459541 ],
       [ 0.18836027, -0.10156041,  0.06029512, -0.01909847,  0.05154379,
        -0.13115072, -0.02483503, -0.01637593],
       [ 0.03252333, -0.15701973, -0.09855079,  0.02527443,  0.11696623,
        -0.11808375,  0.00922776, -0.18057323],
       [-0.00422834, -0.15385334,  0.08371526,  0.04801592, -0.0985806 ,
        -0.15267137,  0.02812213,  0.02258318],
       [ 0.04930976, -0.04624599, -0.09318331, -0.08836022, -0.01470876,
        -0.06025875, -0.2239221 , -0.0299184 ],
       [ 0.05051459, -0.0096743 ,  0.0571319 , -0.01590223,  0.02339932,
        -0.01718113,  0.11465729, -0.14800878],
       [-0.00130149,  0.03694223,  0.09181793,  0.034801  , -0.16368681,
        -0.10045813, -0.1498202 ,  0.07188159],
       [ 0.01939805, -0.01990664, -0.02249878,  0.00197748,  0.06772549,
        -0.12344734,  0.0713622 ,  0.00286786],
       [ 0.861185  , -0.14879651, -0.04108535,  0.00942284, -0.07520685,
        -0.00089158,  0.1909989 ,  0.13913456],
       [-0.00436789,  0.09107946, -0.1586693 , -0.073404  ,  0.06058103,
        -0.07270882,  0.13633692, -0.13203673],
       [ 0.04281171,  0.36552665, -0.12302189, -0.03762249, -0.22615564,
         0.18077435, -0.12230088,  0.02652103],
       [ 0.10940611, -0.3029895 , -0.06101272, -0.07429527, -0.03852937,
         0.13277219, -0.13314696, -0.14582665],
       [-0.11558246,  0.00798289, -0.09431767, -0.23069185, -0.10505641,
        -0.00740103,  0.04360008,  0.1291486 ],
       [ 0.05988555, -0.14522938, -0.02605254, -0.11276164, -0.21254602,
         0.11339637,  0.00127609, -0.03422871],
       [ 0.12860705, -0.04377222, -0.08297722,  0.18541121,  0.04574784,
        -0.00439808, -0.05108613, -0.14495628],
       [ 0.0656451 ,  0.05514231,  0.08627539,  0.0079679 ,  0.13917911,
        -0.06071352,  0.02107348, -0.02480175]], dtype=float32), array([[ 0.9468299 ],
       [ 0.9203023 ],
       [-0.17628585],
       [-0.19782181],
       [-0.15558146],
       [ 0.08695479],
       [-0.07073084],
       [-0.02848704]], dtype=float32)]
>>> create_weights()
[array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 2., 0., 0., 0., 0., 0.],
       [0., 1., 0., 2., 0., 0., 0., 0.]]), array([[1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]]), array([[0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.09531, 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [1.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.5    , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ],
       [0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        0.     ]]), array([[1.],
       [1.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.]])]
>>> model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 6s - loss: 220.9386 - val_loss: 218.5639
Epoch 2/50
 - 2s - loss: 218.1702 - val_loss: 217.8226
Epoch 3/50
 - 1s - loss: 217.9909 - val_loss: 218.0740
Epoch 4/50
 - 1s - loss: 217.9267 - val_loss: 217.7522
Epoch 5/50
 - 1s - loss: 217.7822 - val_loss: 217.7965
Epoch 6/50
 - 1s - loss: 217.7914 - val_loss: 217.6111
Epoch 7/50
 - 1s - loss: 217.8175 - val_loss: 217.8453
Epoch 8/50
 - 1s - loss: 217.7765 - val_loss: 217.8394
Epoch 9/50
 - 1s - loss: 217.7579 - val_loss: 217.8455
Epoch 10/50
 - 1s - loss: 217.7772 - val_loss: 217.7775
Epoch 11/50
 - 1s - loss: 217.7767 - val_loss: 217.9312
Epoch 12/50
 - 1s - loss: 217.7653 - val_loss: 217.8739
Epoch 13/50
 - 1s - loss: 217.7671 - val_loss: 217.9034
Epoch 14/50
 - 1s - loss: 217.7636 - val_loss: 217.3981
Epoch 15/50
Traceback (most recent call last):
  File "<pyshell#768>", line 1, in <module>
    model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> model.get_weights()
[array([[-2.0100544e-03,  2.3604068e-03, -2.6408820e-03,  6.8841706e-04,
        -1.2006625e-03,  1.1939518e-03,  1.6674981e-05,  3.0946231e-03],
       [ 1.8885663e-03, -1.3564654e-03,  2.4504114e-03,  3.2284996e-04,
         1.6459884e-03, -3.9736027e-04, -4.3776413e-04,  2.9337651e-03],
       [ 9.8293358e-01, -2.9370817e-03,  1.9784993e+00,  1.2211091e-03,
        -1.7738712e-03,  2.8880865e-03, -3.1456265e-03, -3.0352867e-03],
       [-2.2012601e-03,  4.9875013e-04,  1.2406245e-03,  1.9981675e+00,
         8.2791416e-04,  1.3787686e-03, -2.2287259e-03, -2.8255363e-03]],
      dtype=float32), array([[ 1.06566966e+00, -1.68786652e-03,  1.53972721e-03,
         1.38207199e-03, -1.43749337e-03,  5.62234200e-05,
         2.04710965e-03,  2.80765607e-03],
       [-5.35912346e-04,  1.71886338e-03, -9.93668800e-04,
        -1.72910048e-03, -1.02043338e-03,  3.95343988e-04,
        -2.57057929e-03, -3.89587338e-04],
       [-2.63714464e-03, -9.22150328e-04,  7.13645935e-01,
         1.96736655e-03, -1.77726056e-03,  1.23248051e-03,
        -2.71420344e-03,  6.33802731e-04],
       [ 1.04051770e-03,  8.82648746e-04,  7.13169396e-01,
         2.73232465e-03, -7.37889030e-04,  2.61612236e-04,
         6.63386425e-04,  7.68038852e-04],
       [-1.66783726e-03,  2.58612284e-03, -1.14813598e-03,
         2.72787758e-03, -2.27206806e-03,  6.28989597e-04,
         2.84668524e-03, -2.09047087e-03],
       [ 1.85293937e-03,  1.29745377e-03, -1.72793109e-04,
        -2.52518547e-03,  2.61398661e-03,  5.91215910e-04,
        -2.79696961e-03,  1.48172677e-03],
       [ 2.27390067e-03, -2.46395101e-03,  2.11949158e-03,
        -1.97463227e-03, -2.80904816e-03, -3.21905420e-04,
        -1.71278394e-03, -2.61652214e-03],
       [-1.77528453e-03,  3.15813138e-03, -1.76065939e-03,
         2.82590953e-03,  3.07795685e-03, -1.54807698e-03,
        -1.86046993e-04, -1.11627334e-04]], dtype=float32), array([[-2.5098403e-03, -1.3489247e-03,  1.3706645e-03,  1.2188726e-03,
         5.2677345e-04, -1.1723721e-03, -2.7472400e-03, -1.8821892e-03],
       [-1.8342575e-03,  6.3026953e-04,  3.3954222e-04, -3.1096905e-03,
         1.9371529e-03,  2.6858165e-03, -7.8972545e-04, -2.9925699e-03],
       [-3.0080345e-03,  1.1920147e-03, -1.3496082e-03, -1.1270596e-03,
         3.0541448e-03,  2.8151739e-03, -2.8767660e-03,  2.5969092e-04],
       [ 2.6146288e-03, -7.1953487e-04, -1.2443600e-03,  2.7947575e-03,
        -3.1210696e-03,  2.1112261e-03,  2.7640136e-03,  2.3467573e-03],
       [-3.7191008e-04,  1.4218733e-03, -1.0955749e-03, -1.1974499e-03,
         2.2169531e-03, -1.0031584e-03, -1.6341599e-03,  2.6788369e-03],
       [-1.0230148e-03,  2.4241540e-03, -2.2370883e-03, -2.9834956e-03,
         2.7322737e-03, -1.4685767e-03, -2.4640569e-03,  2.0604776e-03],
       [ 4.3239602e-04, -2.6326424e-03, -1.7189742e-03, -3.0480467e-03,
        -1.7268345e-03,  1.5615334e-03,  3.6010635e-05,  1.5381919e-03],
       [ 9.5302070e-04, -2.3151417e-03, -3.1306113e-03,  1.3958337e-04,
        -2.7513774e-03,  1.7120086e-03, -2.7832449e-03, -2.2014955e-03],
       [ 9.9810326e-01, -2.7833714e-03,  1.0784614e-03,  2.3606580e-03,
        -2.5742833e-04,  2.7650357e-03,  9.8019920e-04, -1.7813152e-03],
       [ 6.5562502e-04,  1.3384382e-03,  2.1211887e-03, -1.6043603e-03,
        -3.0998616e-03,  6.1197690e-04, -3.0505403e-03,  2.0101925e-03],
       [-1.5714417e-03,  4.8282972e-01,  2.9801093e-03,  2.0493830e-03,
         2.8493728e-03,  1.2226527e-03, -3.1475318e-03, -1.3967576e-03],
       [ 3.1284820e-03,  9.1723376e-04, -2.8826874e-03, -1.2994254e-03,
         9.3534356e-05,  2.0821937e-03, -1.9603199e-03,  2.4387129e-03],
       [ 1.0394576e-03,  1.9309259e-03, -6.3373923e-04, -2.5978051e-03,
         2.5937362e-03, -6.7857094e-04,  2.2192136e-03, -1.4419103e-03],
       [ 3.0101351e-03,  2.0501767e-03,  1.2346701e-03,  3.1416561e-03,
        -6.6506205e-04,  7.5309409e-04, -2.9407269e-03,  1.6690638e-03],
       [-5.6746346e-04, -3.0391132e-03,  3.0341658e-03,  1.0424912e-03,
        -3.0225408e-03, -1.8086027e-03, -1.9782307e-03,  1.4551091e-03],
       [ 2.2297334e-03, -2.1720284e-03, -1.0190128e-03,  1.2721550e-03,
         6.0818915e-05, -5.8977026e-04, -9.9188020e-04, -1.7208920e-04]],
      dtype=float32), array([[ 9.6594369e-01],
       [ 1.1594849e+00],
       [ 1.5647816e-03],
       [-3.0278612e-03],
       [ 2.3079277e-03],
       [-1.1167877e-03],
       [ 2.4793765e-03],
       [ 4.3611208e-04]], dtype=float32)]
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)/10 for w in create_weights()])
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> model.get_weights()
[array([[-2.2180860e-03, -3.9193947e-03,  7.0228579e-04,  3.6231097e-04,
         3.4094451e-03,  6.7975896e-04, -1.3359473e-02, -7.6287086e-03],
       [-1.3219078e-02, -2.9325470e-02,  5.4185586e-03,  9.4928924e-04,
        -2.4817143e-02,  7.3957080e-03, -2.4361975e-02, -3.8619980e-03],
       [ 9.9333912e-01,  1.0048503e+00,  1.9984554e+00, -1.1924788e-02,
         6.9555826e-03, -1.8987710e-02,  1.0895424e-02,  4.7801230e-03],
       [-6.0832002e-03,  9.9206883e-01, -1.3165204e-02,  1.9920170e+00,
        -4.9632452e-03,  1.4472660e-03, -1.1913320e-02,  4.9582636e-03]],
      dtype=float32), array([[ 9.77948189e-01, -2.08099447e-02,  2.39324814e-04,
        -1.38342066e-03, -6.85247872e-03, -5.76464646e-03,
        -6.02383399e-03, -2.77694548e-03],
       [-1.51442562e-03,  1.01254535e+00, -1.71693750e-02,
        -6.24965271e-03,  2.15418916e-03, -3.39046563e-03,
         6.58723386e-03,  2.93670665e-03],
       [-1.35443630e-02,  3.60083254e-03,  9.93369401e-01,
         4.27211914e-03, -3.37957684e-03, -4.06074570e-03,
        -1.42210815e-02,  2.66612135e-03],
       [-1.43033024e-02, -9.88194533e-03,  9.87386584e-01,
        -1.53113771e-02, -2.10593902e-02,  6.29444513e-03,
        -1.94383934e-02, -4.76053497e-03],
       [ 1.40731735e-02, -5.38009871e-03, -1.39862243e-02,
        -3.55430273e-03, -5.34078199e-03, -1.23870131e-02,
        -9.16894351e-04, -3.29053886e-02],
       [-1.65847950e-02,  4.75349184e-03,  8.45931936e-03,
         1.21796569e-02, -2.05137767e-02, -2.45430470e-02,
         7.46233342e-03,  5.99438418e-03],
       [-1.21053569e-02,  1.03205275e-02, -5.00854058e-03,
         9.41554550e-04, -1.08013963e-02,  1.65938754e-02,
        -4.20555286e-03, -7.24387635e-03],
       [ 4.48338455e-04,  1.13227323e-03, -7.48281227e-03,
        -8.30808934e-03, -3.19911614e-02, -1.44593157e-02,
         6.53303321e-03,  3.76086566e-03]], dtype=float32), array([[-1.57236606e-02,  7.53463712e-03,  2.29062105e-04,
         1.87550415e-03, -1.08420616e-02,  4.59716702e-03,
        -1.81094487e-03, -6.07251469e-03],
       [ 9.31027085e-02,  1.51928107e-03, -1.48076341e-02,
         3.58750159e-03, -2.29059774e-02,  1.11212314e-03,
         1.86918303e-03, -2.19775178e-03],
       [-8.18148628e-03,  4.98809014e-03, -1.08054699e-02,
        -7.56856799e-03,  5.28986286e-03, -1.08747398e-02,
         6.29221927e-03, -1.57153513e-02],
       [ 3.47629213e-03, -1.14177456e-02, -9.55563318e-03,
         1.05207562e-02, -8.67364835e-03, -4.86847106e-03,
         1.00541045e-03, -1.11393200e-03],
       [-9.01521533e-04, -6.23020576e-04, -6.51973998e-04,
         6.97444752e-03, -1.00948829e-02, -8.36410746e-03,
         4.25931579e-03, -2.14085840e-02],
       [ 5.24790399e-03, -8.66427831e-03, -8.21781438e-03,
        -3.79132666e-03, -1.44206109e-02, -3.31407017e-03,
        -5.67969168e-03,  2.19692476e-03],
       [-4.53342590e-03, -1.14670219e-02, -1.38526019e-02,
        -1.62248053e-02, -1.38312243e-02, -5.41656744e-03,
         9.07297188e-04, -1.33320130e-02],
       [ 1.14434808e-02,  5.10261860e-04, -4.68380749e-03,
        -1.01701943e-02, -1.00910859e-02, -9.81763937e-03,
         1.11150052e-02, -5.04930224e-03],
       [ 9.94810820e-01, -9.09673329e-03, -1.38424747e-02,
        -1.42922001e-02, -2.17894390e-02, -8.22102651e-03,
        -1.57929156e-02,  1.08743906e-02],
       [-4.64500953e-03, -4.25439142e-03,  5.43406233e-04,
        -2.43296046e-02, -1.20226787e-02,  7.83796515e-03,
        -6.13831077e-03,  4.38785739e-03],
       [-4.82478458e-03,  5.00289857e-01, -2.45559402e-02,
        -2.96914931e-02, -1.54286390e-02, -1.66650768e-02,
        -1.45577304e-02, -1.03341304e-02],
       [ 1.77162001e-04,  1.79522717e-03, -5.68319671e-03,
        -2.69352458e-02,  1.55321939e-03, -1.16102388e-02,
        -1.10822506e-02, -5.09736175e-03],
       [ 2.17161234e-03,  1.05895230e-03, -8.04462470e-03,
        -4.27586539e-03, -1.48283895e-02,  1.54067967e-02,
        -8.68263096e-03, -1.33358892e-02],
       [ 9.46984626e-03, -2.30784584e-02, -5.38566336e-03,
         5.19966066e-04,  4.23531746e-03, -1.92000847e-02,
        -1.55718438e-02, -1.96850616e-02],
       [ 1.68438219e-02,  5.60467644e-03, -3.37147946e-03,
        -1.55971367e-02,  2.88841850e-03, -1.08159734e-02,
         1.03180343e-02, -1.09712463e-02],
       [ 2.69722217e-03,  1.74122013e-03, -1.02686901e-02,
        -9.67236981e-03, -9.88692045e-03, -9.06316563e-03,
        -9.20832274e-04,  1.13223074e-02]], dtype=float32), array([[ 0.99672437],
       [ 0.977517  ],
       [-0.00278026],
       [-0.02275671],
       [ 0.00389624],
       [-0.00792316],
       [-0.01914107],
       [-0.00127547]], dtype=float32)]
>>> model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 2s - loss: 218.6132 - val_loss: 217.9606
Epoch 2/50
 - 2s - loss: 217.9462 - val_loss: 217.9778
Epoch 3/50
 - 1s - loss: 217.9752 - val_loss: 218.0098
Epoch 4/50
 - 1s - loss: 217.9446 - val_loss: 218.0259
Epoch 5/50
 - 1s - loss: 217.8378 - val_loss: 217.8394
Epoch 6/50
 - 1s - loss: 217.7496 - val_loss: 218.0103
Epoch 7/50
 - 1s - loss: 217.7755 - val_loss: 217.7040
Epoch 8/50
 - 1s - loss: 217.7714 - val_loss: 217.8676
Epoch 9/50
Traceback (most recent call last):
  File "<pyshell#773>", line 1, in <module>
    model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> model.get_weights()
[array([[-2.0671200e-03,  3.1296758e-03,  5.6381844e-04,  3.0214756e-03,
        -1.2657825e-03,  1.3188061e-03, -2.1936246e-03,  2.0180109e-03],
       [ 1.6260414e-03, -2.3332632e-03,  5.2835175e-04,  1.1204281e-03,
         1.3929746e-03,  1.7767797e-03, -2.6906251e-03,  1.2562668e-03],
       [ 9.9258685e-01, -1.3790320e-03,  1.9804183e+00,  3.1398695e-03,
         2.5653441e-03,  1.1783813e-03, -2.7898391e-04, -3.0658126e-03],
       [ 2.8270686e-03, -1.2498309e-03,  3.1440314e-03,  2.0043590e+00,
         3.0264482e-03,  2.2885397e-03, -2.8107380e-03, -2.4686118e-03]],
      dtype=float32), array([[ 9.8907417e-01,  3.0870906e-03,  2.6470346e-03,  2.3589381e-03,
         1.8094189e-03,  5.7091808e-04,  1.6529978e-03,  3.1171248e-03],
       [ 1.7579743e-03, -1.6211049e-03,  2.2820802e-03, -2.8009147e-03,
        -3.7372718e-04,  2.2268377e-03, -2.4067180e-03,  2.9820325e-03],
       [ 1.8482609e-04, -2.2893492e-03,  7.9201019e-01, -2.8448973e-03,
         3.0321300e-03,  5.2144896e-04,  1.5161724e-03, -4.4591230e-04],
       [-2.8557819e-03,  2.4397203e-03,  7.9509437e-01, -1.6567257e-03,
        -1.0008777e-03,  2.8688301e-04,  1.9455678e-03, -1.9759550e-03],
       [-3.0795068e-03, -2.8028284e-04, -3.0593935e-03,  2.5106454e-03,
        -3.4419545e-03, -4.9786019e-04,  1.6255220e-03, -2.8261601e-03],
       [ 2.6327390e-03,  8.5464562e-06, -2.2737263e-03,  2.4662833e-03,
         6.7779678e-04,  1.3348248e-03, -2.1795428e-03,  3.1039477e-03],
       [ 3.0734837e-03,  1.8661955e-03, -2.3524906e-03, -3.1202261e-03,
        -2.2572004e-03,  2.5280111e-03, -8.8242465e-04, -4.8705458e-04],
       [-2.5861552e-03, -1.4152025e-03, -1.7874897e-04,  2.7026490e-03,
        -2.7984600e-03,  2.5678806e-03, -1.6845961e-03, -8.0126419e-04]],
      dtype=float32), array([[ 1.7241846e-03, -2.4195847e-03, -3.0756404e-03, -3.1385138e-03,
        -8.3112973e-05,  1.1124657e-03, -1.3900229e-03,  2.9652135e-03],
       [ 2.1702531e-03,  3.0854084e-03, -1.4126034e-03, -9.9259696e-04,
         2.7717138e-03, -1.9173683e-03, -8.5333287e-04, -2.9889849e-04],
       [ 2.4890639e-03, -1.7908269e-03,  7.8907411e-05,  2.6337714e-03,
        -4.9501332e-04, -1.0330270e-03,  8.3959562e-04, -8.3712453e-04],
       [ 2.7053747e-03,  2.5373893e-03,  2.1836513e-03,  1.2017738e-03,
         2.2564717e-03,  3.0150907e-03, -1.3319633e-03,  9.9910982e-04],
       [ 2.2363507e-03, -2.6190272e-03, -6.1411818e-05, -2.7710365e-03,
         2.8369040e-03,  3.1614620e-03, -1.3833302e-03,  1.9214866e-03],
       [-3.0530822e-03, -1.9275099e-03, -7.8762218e-04, -2.3109878e-03,
        -1.0069216e-03, -2.1434056e-03,  1.0719475e-03,  5.7982304e-04],
       [-2.3882147e-03, -2.7251788e-03, -5.9330522e-04, -2.3617054e-04,
         1.9747459e-03,  2.6298307e-03,  3.0247639e-03, -1.4955638e-03],
       [ 3.0954015e-03, -2.9517615e-03,  1.8037561e-03,  2.1288639e-03,
         2.4696048e-03, -1.7326430e-04, -1.8398169e-03, -2.8712470e-03],
       [ 9.9955416e-01,  1.1032412e-03,  1.9180673e-03,  2.3337987e-03,
        -2.7972462e-03,  2.3464689e-03,  3.1581556e-05,  1.1451868e-03],
       [-2.5756343e-03, -2.6648045e-03, -1.7859802e-03, -8.6814858e-04,
        -5.1303650e-05,  1.7315425e-03,  3.0962063e-03, -2.0843886e-03],
       [-2.5835100e-03,  5.0392795e-01, -1.8601430e-03,  4.4975564e-04,
        -3.0975847e-03,  2.8709355e-03,  3.0007623e-03,  2.7078060e-03],
       [ 2.1611189e-03, -6.9125497e-04, -2.6494183e-04, -1.7556010e-03,
        -2.8210531e-03, -2.3843592e-03,  8.5916824e-04, -1.6206964e-03],
       [ 3.1089410e-03, -2.8459101e-03, -2.7605752e-03,  3.1571232e-03,
        -3.0337118e-03,  1.8770603e-03, -2.8619973e-03,  4.3686334e-04],
       [ 8.3058875e-04, -3.6613649e-04,  1.2552147e-03, -1.6539720e-03,
         2.9474853e-03, -2.4491795e-03, -2.8984041e-03,  2.5312258e-03],
       [ 2.7730442e-03,  1.0125177e-03,  9.8616839e-04, -9.4821339e-04,
         1.7811459e-03,  2.7896729e-03, -1.7179401e-03,  2.9065732e-03],
       [-3.1606322e-03, -2.7875619e-03,  1.8432669e-03, -2.9687532e-03,
         2.1686015e-04,  2.9072396e-03, -2.7371787e-03, -1.0149328e-03]],
      dtype=float32), array([[ 1.0046923e+00],
       [ 1.0769030e+00],
       [-3.0974117e-03],
       [ 2.3995251e-03],
       [ 2.1968519e-03],
       [ 1.5078955e-03],
       [ 5.7241460e-04],
       [-4.2497829e-04]], dtype=float32)]
>>> weights0 = model.get_weights()
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 4s - loss: 217.8167 - val_loss: 217.7361
<keras.callbacks.callbacks.History object at 0x000001CFAABE0DA0>
>>> model.get_weights()
[array([[-3.1079792e-03, -9.5049792e-05,  2.9853552e-03,  3.0804174e-03,
         2.9989544e-03,  3.0893837e-03, -1.8293600e-03,  1.9379823e-03],
       [-2.5014363e-03, -3.0487122e-03, -1.7888646e-04, -1.9782019e-04,
        -1.8024641e-03, -1.7559089e-03,  1.4741651e-03, -4.6112182e-04],
       [ 9.9413818e-01,  1.0758551e-04,  1.9814368e+00,  6.7347958e-04,
         2.9945420e-03,  2.9837941e-03, -1.5399558e-04, -3.4425524e-05],
       [ 1.7618211e-03, -2.9061632e-03, -9.1001100e-04,  2.0015078e+00,
        -2.0982791e-03, -2.3141515e-03,  3.0620070e-03, -4.8609197e-04]],
      dtype=float32), array([[ 9.8526764e-01, -2.7028753e-03, -2.6502099e-03,  2.0189509e-03,
         1.7240065e-03, -2.2908279e-03,  3.0639305e-04, -3.8235681e-05],
       [ 1.6153485e-03,  1.5826294e-03,  3.5203822e-04,  1.2770137e-03,
        -8.0033217e-04, -1.7325467e-03, -3.0876824e-03,  9.9502341e-04],
       [-1.1751844e-03,  3.0796144e-03,  7.7141613e-01,  2.7875437e-03,
        -2.3488812e-03,  2.2584680e-03,  1.2945510e-03, -2.7206319e-03],
       [-1.8391085e-03,  3.0841662e-03,  7.8480709e-01,  1.8119530e-03,
         2.5382547e-03,  4.7963465e-04,  3.0744963e-03, -2.9310738e-03],
       [ 3.1198349e-03, -2.2777093e-03,  1.8807497e-03, -2.9052049e-03,
         2.3506354e-03, -5.4620794e-04,  1.0750974e-03, -2.6030261e-03],
       [-9.0133864e-04, -7.8939437e-04,  3.0804214e-03,  1.3617494e-03,
        -6.1803684e-04, -2.2290398e-03, -2.0557970e-03, -2.7344809e-03],
       [-3.0284505e-03,  2.9712401e-03,  2.2415582e-03,  2.8865705e-03,
        -2.9545624e-03,  2.0547775e-03, -6.8602531e-04,  5.1302742e-04],
       [-1.0858966e-03, -1.2373752e-03, -3.9067245e-04, -1.3215236e-03,
         3.0712152e-03, -1.2776906e-03, -2.5803936e-03,  2.7070069e-03]],
      dtype=float32), array([[ 1.7845246e-04,  7.6890911e-04, -3.1157036e-03, -1.8486766e-03,
        -1.3839969e-03,  1.6839446e-03, -3.9566675e-04,  1.8570649e-03],
       [-1.9899353e-03,  1.8354389e-03, -1.1952764e-03, -1.1266704e-04,
         1.3810868e-03, -1.8321379e-03,  4.2184722e-04, -1.5657465e-04],
       [-2.0092081e-03,  2.8474317e-03, -1.8723528e-03,  1.4658177e-03,
         3.1127213e-03, -2.9784197e-03, -2.6680357e-03, -8.3204079e-04],
       [ 2.9820669e-03,  1.3000215e-03, -1.0327835e-03, -3.1094044e-04,
         1.4612022e-03, -2.5176122e-03,  9.1110048e-04,  7.2804815e-04],
       [-1.8675337e-03, -3.0594382e-03,  1.0648506e-03, -8.3028467e-04,
        -1.2417813e-04, -1.1982971e-03, -3.1576951e-03, -4.5410925e-04],
       [ 9.3014212e-05,  1.9793096e-03,  1.9933872e-03, -2.2831270e-03,
         2.9633916e-03,  2.1657022e-04, -4.3532578e-05, -6.9760537e-04],
       [-2.4605652e-03, -1.3518647e-03, -1.7913196e-03, -1.4325231e-03,
        -2.2809508e-03, -2.5307182e-03,  4.0951971e-04,  1.2766814e-04],
       [-2.0294674e-03, -8.2288653e-04, -2.9585301e-03, -6.0533627e-04,
        -1.5139651e-03, -7.5354613e-04, -2.9678294e-03, -2.3854743e-03],
       [ 1.0008929e+00,  3.1460423e-03,  8.6392043e-05, -2.5258786e-03,
        -2.6494339e-03, -3.4681158e-04,  9.0449268e-04,  2.3474381e-03],
       [-5.3772097e-04, -2.8733187e-03, -1.4315450e-03, -2.2109977e-03,
        -2.9677426e-04,  2.6895567e-03, -2.8624211e-03,  3.1148484e-03],
       [-1.8020419e-03,  5.0542098e-01,  2.5298905e-03, -1.5023069e-03,
         3.1268527e-03, -1.0349189e-03, -2.9529242e-03,  1.7606596e-03],
       [ 3.1560380e-03,  2.4753606e-03, -2.8681590e-03,  7.9735601e-04,
         2.7276862e-03,  1.2964351e-03, -9.4904541e-04, -4.6734407e-04],
       [-9.7792002e-04, -4.8041396e-04,  9.9141407e-04,  2.4420409e-03,
         1.7428399e-03,  2.9967171e-03, -1.1975234e-03, -7.5274904e-04],
       [ 2.5810304e-04,  1.1819468e-03, -1.9831159e-03,  1.3566385e-03,
        -2.2279809e-03, -1.9734732e-03,  1.7472783e-03,  2.5582097e-03],
       [ 1.8198962e-03, -5.7410903e-04, -9.9688768e-04, -8.1740873e-04,
        -2.7207003e-03,  2.2791934e-03,  2.3491965e-03,  2.8740154e-03],
       [ 1.4853518e-03, -1.8449631e-03,  3.0870545e-03,  3.1442905e-04,
        -2.3091089e-03, -2.8199879e-03, -2.6440946e-03,  2.9649708e-04]],
      dtype=float32), array([[ 1.0008125e+00],
       [ 1.1020578e+00],
       [ 1.9238604e-04],
       [ 3.1484473e-03],
       [-6.9100282e-04],
       [-9.6004590e-04],
       [-1.0803784e-03],
       [ 2.7731545e-03]], dtype=float32)]
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 5s - loss: nan - val_loss: nan
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
 - 1s - loss: nan - val_loss: nan
Epoch 4/50
Traceback (most recent call last):
  File "<pyshell#779>", line 1, in <module>
    model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> y_pred=model.predict(x_train)
>>> model.losses
[<tf.Tensor 'power_dense_1/weight_regularizer/Cast:0' shape=() dtype=float64>, <tf.Tensor 'dense_1/weight_regularizer/Cast:0' shape=() dtype=float64>, <tf.Tensor 'power_dense_2/weight_regularizer/Cast:0' shape=() dtype=float64>, <tf.Tensor 'dense_2/weight_regularizer/Cast:0' shape=() dtype=float64>]
>>> complex_mean_absolute_error(y_train, y_pred)
<tf.Tensor: id=8824, shape=(), dtype=float64, numpy=0.05135682872311349>
>>> [l0_1(w) for w in model.get_weights()]
[<tf.Tensor: id=8840, shape=(), dtype=float64, numpy=30.66754150390625>, <tf.Tensor: id=8847, shape=(), dtype=float64, numpy=60.88429260253906>, <tf.Tensor: id=8854, shape=(), dtype=float64, numpy=121.69508361816406>, <tf.Tensor: id=8861, shape=(), dtype=float64, numpy=7.683004856109619>]
>>> model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: nan - val_loss: nan
Epoch 2/50
Traceback (most recent call last):
  File "<pyshell#785>", line 1, in <module>
    model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 210, in fit_loop
    verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 449, in test_loop
    batch_outs = f(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> y_pred=model.predict(x_train)
>>> model.loss_weights
>>> model.loss_weights()
Traceback (most recent call last):
  File "<pyshell#789>", line 1, in <module>
    model.loss_weights()
TypeError: 'NoneType' object is not callable
>>> model.evaluate(x_train,y_train)
  32/8000 [..............................] - ETA: 1s  64/8000 [..............................] - ETA: 28s  96/8000 [..............................] - ETA: 26s 128/8000 [..............................] - ETA: 24s 160/8000 [..............................] - ETA: 25s 192/8000 [..............................] - ETA: 24s 224/8000 [..............................] - ETA: 24s 256/8000 [..............................] - ETA: 25s 288/8000 [>.............................] - ETA: 26s 320/8000 [>.............................] - ETA: 27s 352/8000 [>.............................] - ETA: 28s 384/8000 [>.............................] - ETA: 29s 416/8000 [>.............................] - ETA: 30s 448/8000 [>.............................] - ETA: 31s 480/8000 [>.............................] - ETA: 33s 512/8000 [>.............................] - ETA: 34s 544/8000 [=>............................] - ETA: 35s 576/8000 [=>............................] - ETA: 36s 608/8000 [=>............................] - ETA: 37s 640/8000 [=>............................] - ETA: 38s 672/8000 [=>............................] - ETA: 38s 704/8000 [=>............................] - ETA: 39sTraceback (most recent call last):
  File "<pyshell#790>", line 1, in <module>
    model.evaluate(x_train,y_train)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1361, in evaluate
    callbacks=callbacks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 468, in test_loop
    progbar.update(batch_end)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\generic_utils.py", line 385, in update
    sys.stdout.write('\b' * prev_total_width)
KeyboardInterrupt
>>> model.evaluate(x_train,y_train,verbose=0)
221.5042584783733
>>> model.count_params()
232
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 3s - loss: nan - val_loss: nan
<keras.callbacks.callbacks.History object at 0x000001C6683D5550>
>>> model.evaluate(x_train,y_train,verbose=0)
nan
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> model.add_metric(complex_mean_absolute_error)
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 559, in make_tensor_proto
    str_values = [compat.as_bytes(x) for x in proto_values]
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 559, in <listcomp>
    str_values = [compat.as_bytes(x) for x in proto_values]
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\compat.py", line 65, in as_bytes
    (bytes_or_text,))
TypeError: Expected binary or unicode string, got <function complex_mean_absolute_error at 0x000001C665666598>

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#796>", line 1, in <module>
    model.add_metric(complex_mean_absolute_error)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 989, in add_metric
    metric_obj = _create_mean_metric(value, name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 1251, in _create_mean_metric
    _call_metric(metric_obj, value)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 75, in symbolic_fn_wrapper
    return func(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\base_layer.py", line 1257, in _call_metric
    update_op = metric_obj.update_state(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\metrics_utils.py", line 42, in decorated
    update_op = update_state_fn(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\metrics.py", line 172, in update_state
    values = K.cast(values, self.dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 1239, in cast
    return tf.cast(x, dtype)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 687, in cast
    x = ops.convert_to_tensor(x, name="x")
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1100, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1158, in convert_to_tensor_v2
    as_ref=False)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\ops.py", line 1237, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 305, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 246, in constant
    allow_broadcast=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\constant_op.py", line 284, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\tensor_util.py", line 563, in make_tensor_proto
    "supported type." % (type(values), values))
TypeError: Failed to convert object of type <class 'function'> to Tensor. Contents: <function complex_mean_absolute_error at 0x000001C665666598>. Consider casting elements to a supported type.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 167, in <module>
    model.evaluate(x_test,y_test,verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1361, in evaluate
    callbacks=callbacks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 449, in test_loop
    batch_outs = f(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_absolute_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_576]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 167, in <module>
    model.evaluate(x_test,y_test,verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1361, in evaluate
    callbacks=callbacks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 449, in test_loop
    batch_outs = f(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_absolute_error/Sum (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_572]

Function call stack:
keras_scratch_graph

>>> x = tf.constant([[1., 1.], [2., 2.]])
>>> x
<tf.Tensor: id=578, shape=(2, 2), dtype=float32, numpy=
array([[1., 1.],
       [2., 2.]], dtype=float32)>
>>> tf.reduce_mean(x)
<tf.Tensor: id=581, shape=(), dtype=float32, numpy=1.5>
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
>>> model.evaluate(x_test,complex_target(y_test),verbose=0)
Traceback (most recent call last):
  File "<pyshell#800>", line 1, in <module>
    model.evaluate(x_test,complex_target(y_test),verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1361, in evaluate
    callbacks=callbacks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 372, in test_loop
    steps_name='steps')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 571, in check_num_samples
    'you should specify the `' + steps_name + '` argument '
ValueError: If your data is in the form of symbolic tensors, you should specify the `steps` argument (instead of the `batch_size` argument, because symbolic tensors are expected to produce batches of input data).
>>> y_test = complex_target(y_test)
>>> y_test.shape
TensorShape([2000, 1, 2])
>>> model.evaluate(x_test,y_test,verbose=0)
Traceback (most recent call last):
  File "<pyshell#803>", line 1, in <module>
    model.evaluate(x_test,y_test,verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1361, in evaluate
    callbacks=callbacks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 372, in test_loop
    steps_name='steps')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 571, in check_num_samples
    'you should specify the `' + steps_name + '` argument '
ValueError: If your data is in the form of symbolic tensors, you should specify the `steps` argument (instead of the `batch_size` argument, because symbolic tensors are expected to produce batches of input data).
>>> model.evaluate(x_test,y_test,verbose=0,steps=32)
[6.930891738273203, 4.967281341552734]
>>> y_pred = model.predict(x_test)
>>> complex_mean_absolute_error(y_test, y_pred)
<tf.Tensor: id=1585, shape=(), dtype=float64, numpy=0.465617671189634>
>>> model.metrics
[<keras.metrics.MeanMetricWrapper object at 0x000001F4177894A8>]
>>> model = Model(inputs=[value_in], outputs=[output])
>>> model.compile(loss=complex_absolute_error,optimizer="rmsprop", metrics=[complex_mean_absolute_error])
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> model.evaluate(x_test,y_test)
Traceback (most recent call last):
  File "<pyshell#811>", line 1, in <module>
    model.evaluate(x_test,y_test)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1361, in evaluate
    callbacks=callbacks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 372, in test_loop
    steps_name='steps')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 571, in check_num_samples
    'you should specify the `' + steps_name + '` argument '
ValueError: If your data is in the form of symbolic tensors, you should specify the `steps` argument (instead of the `batch_size` argument, because symbolic tensors are expected to produce batches of input data).
>>> model.evaluate(x_test,y_test,verbose=0,steps=32)
[6.924202949739993, 15.089862048625946]
>>> complex_mean_absolute_error(y_test,model.predict(x_test))
<tf.Tensor: id=2476, shape=(), dtype=float64, numpy=0.4715581738154136>
>>> model.evaluate(x_test,y_test,verbose=0,steps=16)
[13.848405899479985, 7.544930070638657]
>>> model.evaluate(x_test,y_test,verbose=0,steps=1)
[221.57449439167976, 0.4715581238269806]
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test),steps=1)
Traceback (most recent call last):
  File "<pyshell#816>", line 1, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test),steps=1)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1118, in fit
    raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
TypeError: Unrecognized keyword arguments: {'steps': 1}
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
Traceback (most recent call last):
  File "<pyshell#817>", line 1, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics_1/complex_mean_absolute_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_4545]

Function call stack:
keras_scratch_graph

>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test),batch_size=32)
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
Traceback (most recent call last):
  File "<pyshell#818>", line 1, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test),batch_size=32)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics_1/complex_mean_absolute_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_4545]

Function call stack:
keras_scratch_graph

>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test),batch_size=1000)
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
Traceback (most recent call last):
  File "<pyshell#819>", line 1, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test),batch_size=1000)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics_1/complex_mean_absolute_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_4545]

Function call stack:
keras_scratch_graph

>>> model.loss
<function complex_absolute_error at 0x000001F417676488>
>>> model.metrics
[<keras.metrics.MeanMetricWrapper object at 0x000001F418C00C50>]
>>> model.metrics[0]
<keras.metrics.MeanMetricWrapper object at 0x000001F418C00C50>
>>> model.metrics[0]
<keras.metrics.MeanMetricWrapper object at 0x000001F418C00C50>
>>> model.metrics[0]._fn
<function complex_mean_absolute_error at 0x000001F417676598>
>>> model.metrics[0]._fn_kwargs
{}
>>> model.metrics[0]._fn(y_test,model.predict(x_test))
<tf.Tensor: id=4756, shape=(), dtype=float64, numpy=0.382567852696286>
>>> model.metrics[0].call(y_test,model.predict(x_test))
Traceback (most recent call last):
  File "<pyshell#827>", line 1, in <module>
    model.metrics[0].call(y_test,model.predict(x_test))
TypeError: call() takes 2 positional arguments but 3 were given
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
(None, None, None) (None, 1, 2)
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 170, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 196, in fit_loop
    outs = fit_function(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Invalid reduction dimension (3 for input with 3 dimension(s)
	 [[node metrics/complex_mean_absolute_error/Mean (defined at C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_2298]

Function call stack:
keras_scratch_graph

>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 170, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1213, in fit
    self._make_train_function()
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 333, in _make_train_function
    **self._function_kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\backend\tensorflow_backend.py", line 3009, in function
    **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3543, in function
    'eager execution. You passed: %s' % (kwargs,))
ValueError: Session keyword arguments are not support during eager execution. You passed: {'validation_steps': 1}
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 170, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 210, in fit_loop
    verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 372, in test_loop
    steps_name='steps')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 571, in check_num_samples
    'you should specify the `' + steps_name + '` argument '
ValueError: If your data is in the form of symbolic tensors, you should specify the `steps` argument (instead of the `batch_size` argument, because symbolic tensors are expected to produce batches of input data).
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 170, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test), validation_steps=1)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 82, in fit_loop
    raise ValueError('Can only use `validation_steps` '
ValueError: Can only use `validation_steps` when doing step-wise training, i.e. `steps_per_epoch` must be set.
>>> 8000/32
250.0
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 3, in <module>
    from keras.models import Model, Sequential
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\__init__.py", line 3, in <module>
    from . import utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\__init__.py", line 2, in <module>
    from . import np_utils
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\np_utils.py", line 6, in <module>
    import numpy as np
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\__init__.py", line 145, in <module>
    from . import lib
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\lib\__init__.py", line 12, in <module>
    from .nanfunctions import *
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
  File "<frozen importlib._bootstrap_external>", line 764, in get_code
  File "<frozen importlib._bootstrap_external>", line 832, in get_data
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 165, in <module>
    N_t
NameError: name 'N_t' is not defined
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 174, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test), steps_per_epoch=int(Nt/32), validation_steps=1)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 152, in fit_loop
    outs = fit_function(fit_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
Traceback (most recent call last):
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py", line 174, in <module>
    model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))#, steps_per_epoch=int(Nt/32), validation_steps=1
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 210, in fit_loop
    verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 372, in test_loop
    steps_name='steps')
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_utils.py", line 571, in check_num_samples
    'you should specify the `' + steps_name + '` argument '
ValueError: If your data is in the form of symbolic tensors, you should specify the `steps` argument (instead of the `batch_size` argument, because symbolic tensors are expected to produce batches of input data).
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 5s - loss: nan - val_loss: nan
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 5s - loss: 226.0830 - val_loss: 226.8716
>>> model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/50
 - 3s - loss: 228.2894 - val_loss: 227.6613
Epoch 2/50
 - 2s - loss: nan - val_loss: nan
Epoch 3/50
 - 2s - loss: nan - val_loss: nan
Epoch 4/50
 - 2s - loss: nan - val_loss: nan
Epoch 5/50
 - 1s - loss: nan - val_loss: nan
Epoch 6/50
 - 1s - loss: nan - val_loss: nan
Epoch 7/50
 - 1s - loss: nan - val_loss: nan
Epoch 8/50
 - 1s - loss: nan - val_loss: nan
Epoch 9/50
 - 1s - loss: nan - val_loss: nan
Epoch 10/50
 - 1s - loss: nan - val_loss: nan
Epoch 11/50
 - 1s - loss: nan - val_loss: nan
Epoch 12/50
 - 1s - loss: nan - val_loss: nan
Epoch 13/50
 - 1s - loss: nan - val_loss: nan
Epoch 14/50
 - 1s - loss: nan - val_loss: nan
Epoch 15/50
 - 1s - loss: nan - val_loss: nan
Epoch 16/50
Traceback (most recent call last):
  File "<pyshell#829>", line 1, in <module>
    model.fit(x_train,y_train,epochs=50,verbose=2,validation_data=(x_test, y_test))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1239, in fit
    validation_freq=validation_freq)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 210, in fit_loop
    verbose=0)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 449, in test_loop
    batch_outs = f(ins_batch)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 3s - loss: nan - val_loss: nan
<keras.callbacks.callbacks.History object at 0x000002E3B76319E8>
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 3s - loss: nan - val_loss: nan
<keras.callbacks.callbacks.History object at 0x000002E3B7631A90>
>>> sgd = SGD(learning_rate=0.001)
>>> model.compile(loss=complex_absolute_error, optimizer=sgd, metrics=[])
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 5s - loss: 225.7381 - val_loss: 226.9793
<keras.callbacks.callbacks.History object at 0x000002E3B71677F0>
>>> model.get_weights()
[array([[-1.06724396e-01, -1.34047300e-01, -6.28561452e-02,
         1.00570485e-01, -5.08360453e-02, -1.93426326e-01,
        -1.16212247e-02, -3.00220791e-02],
       [ 8.07856489e-03,  1.66867539e-01, -1.64984143e+00,
         9.22024716e-04, -3.11228931e-02, -1.33965924e-01,
        -9.49589610e-02,  9.29732248e-02],
       [ 9.45235610e-01,  9.39500272e-01,  2.02051258e+00,
        -1.70864031e-01,  4.07639951e-01,  9.37027708e-02,
        -5.21824583e-02, -5.80995120e-02],
       [ 2.07376555e-02,  9.72773969e-01, -6.43222593e-04,
         1.99632657e+00, -1.66326866e-01,  6.01461437e-03,
         3.88325751e-01, -1.52489785e-02]], dtype=float32), array([[ 9.80563819e-01,  1.10299781e-01,  6.25196248e-02,
        -2.34161224e-02,  8.30578431e-02, -1.78108647e-01,
        -3.90646487e-01,  1.15610622e-01],
       [ 4.45955896e+00,  1.01694918e+00,  3.41176927e-01,
        -1.17313005e-02, -1.08165527e-02,  2.66870856e-01,
        -6.84832484e-02,  1.27892658e-01],
       [-4.64965031e-02, -3.83458674e-01,  9.86250699e-01,
         6.11357950e-02,  2.75771022e-01, -9.40653419e+00,
         7.88015127e-02,  9.65891704e-02],
       [ 6.30501211e-02,  4.02799428e-01,  9.92788792e-01,
         1.55518085e-01,  1.84883028e-01, -5.11233695e-02,
         3.54595691e-01,  2.08853800e-02],
       [ 1.27841577e-01,  2.31115207e-01,  4.86436859e-02,
         1.06303029e-01, -2.39131957e-01,  1.05265997e-01,
        -7.69473240e-02, -1.56781685e+00],
       [-1.13384612e-02,  2.08197907e-01,  7.85724726e-03,
         1.51230916e-01,  7.51505941e-02,  3.77013162e-02,
         2.15081535e-02, -1.41059804e+00],
       [ 2.50329003e-02,  3.60889643e-01,  7.40311205e-01,
         3.09270266e-02, -4.22160365e-02, -1.82359427e-01,
         8.78547132e-02, -1.51243284e-01],
       [-5.48795462e-02,  3.68933856e-01,  9.10703242e-02,
         1.75490484e-01,  2.09322035e-01,  5.83180087e-03,
        -7.07984418e-02,  6.72697872e-02]], dtype=float32), array([[-1.76742859e-02, -1.06925825e-02, -1.55082904e-02,
         4.39829201e-01, -9.59949791e-02,  5.78498781e-01,
         2.76036039e-02,  9.08106789e-02],
       [ 4.66205850e-02,  4.29731533e-02,  6.96027577e-02,
         2.29964852e+00, -1.68977034e+00,  1.01727314e-01,
        -1.47135884e-01,  9.48793720e-04],
       [ 1.37368962e-02, -8.35323930e-02,  3.64026092e-02,
         2.17005480e-02, -6.33527339e-02, -8.69114324e-02,
         1.69602223e-02, -8.71805474e-03],
       [-1.00423765e+00, -2.32325077e-01,  5.97110391e-02,
        -9.99047514e-03, -4.19171676e-02, -1.52446246e-02,
         6.32659160e-03, -1.58240600e-03],
       [-6.48908973e-01, -5.59757371e-03,  9.30099189e-03,
        -3.07223089e-02,  6.88455641e-01, -1.81276643e+00,
        -4.67691794e-02,  7.05345571e-02],
       [ 1.98300421e-01,  6.42866194e-02,  2.74477094e-01,
         2.59393211e-02,  3.28318655e-01, -2.44674757e-02,
         3.46190408e-02,  9.33654979e-02],
       [ 1.70113876e-01, -2.17685699e-02,  4.55487110e-02,
         3.10057327e-02, -3.89090665e-02,  1.34822400e-02,
        -2.86382467e-01,  9.14499536e-02],
       [ 4.02390957e-01, -7.03614727e-02,  1.21047579e-01,
        -2.43126769e-02,  4.86474819e-02, -3.74923386e-02,
         1.15509182e-02, -7.78005794e-02],
       [ 9.03619707e-01,  1.30250277e-02,  2.92804271e-01,
         1.57412104e-02, -1.93816155e-01,  1.05901086e+00,
         2.03788802e-01,  1.48142070e-01],
       [ 3.20431516e-02,  1.15008064e-01, -2.13040844e-01,
        -5.78645587e-01,  2.17696112e-02, -3.71966101e-02,
         2.28033841e-01,  8.62125009e-02],
       [-3.85599993e-02,  4.52249020e-01, -4.96332422e-02,
         3.56577188e-02, -3.94385792e-02, -1.34222314e-01,
        -9.53700095e-02,  7.89932013e-02],
       [ 2.23445403e-03,  1.19891889e-01, -2.71969084e-02,
        -2.14625150e-02, -7.30127618e-02, -1.76641390e-01,
        -6.11210540e-02, -1.23611302e-03],
       [-1.79880075e-02, -9.69960354e-03, -2.86300868e-01,
         2.48957962e-01,  2.96667349e-02, -1.56139694e-02,
        -1.26369253e-01, -1.15106665e-01],
       [ 1.38982594e-01,  4.75121029e-02, -5.28296046e-02,
         2.23858967e-01,  1.51096344e+00,  1.24723651e-03,
         3.69640626e-02, -2.48110387e-04],
       [ 4.21443675e-03,  2.74772681e-02,  2.35369839e-02,
         9.24968049e-02,  3.56447957e-02, -6.32602870e-02,
        -1.74388196e-02,  2.96540648e-01],
       [ 6.10144697e-02,  2.70896852e-01,  1.01653822e-01,
         3.90046388e-01, -7.86707103e-02, -6.71640038e-01,
         1.03310339e-01,  5.07783964e-02]], dtype=float32), array([[ 0.9710689 ],
       [ 0.9488205 ],
       [-0.04173978],
       [-0.02327912],
       [ 1.1790959 ],
       [-0.10369998],
       [ 0.01412374],
       [-0.06454792]], dtype=float32)]
>>> model0 = model.get_weights()
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 3s - loss: 227.5130 - val_loss: 227.2092
<keras.callbacks.callbacks.History object at 0x000002E3B8FAD0B8>
>>> model1 = model.get_weights()
>>> [m1-m0 for (m1,m0) in np.zip(model1,model0)]
Traceback (most recent call last):
  File "<pyshell#842>", line 1, in <module>
    [m1-m0 for (m1,m0) in np.zip(model1,model0)]
AttributeError: module 'numpy' has no attribute 'zip'
>>> [m1-m0 for (m1,m0) in zip(model1,model0)]
[array([[-3.30846012e-03, -6.12574965e-02,  1.06278956e-01,
        -6.71731234e-02,  5.91097288e-02, -1.26332194e-02,
        -1.25699928e-02,  3.21911871e-02],
       [ 6.98120221e-02, -1.81049004e-01, -3.89361382e-03,
        -1.30837440e-01,  2.81362146e-01,  7.77733326e-03,
         3.37675959e-03, -7.97779560e-02],
       [-8.54659081e-03,  1.81260109e-02, -7.78603554e-03,
         1.30965263e-02,  7.16972351e-03,  2.36606970e-02,
         5.78623414e-02,  4.97206151e-02],
       [ 1.31955028e-01,  1.34406686e-02,  1.33937434e-03,
        -1.19924545e-04,  6.65171966e-02,  6.19498193e-02,
        -3.16617787e-02,  1.48224801e-01]], dtype=float32), array([[-2.48395205e-02, -1.64359808e-01, -7.88191855e-02,
         4.48045731e-02, -6.10124506e-02,  1.45135224e-02,
        -1.25142932e-03, -2.16485932e-02],
       [-1.66358948e-02, -9.04332399e-02, -6.59766793e-03,
         1.81267247e-01,  4.10689861e-02, -8.44880939e-03,
         3.95909622e-02, -1.72521621e-02],
       [ 6.41659349e-02, -4.06542420e-03,  4.29785252e-03,
        -3.90945561e-02,  2.70280242e-03,  1.69563293e-03,
        -1.96746327e-02, -3.23446468e-02],
       [-5.60773909e-02, -9.74488556e-02,  8.76283646e-03,
         2.88522393e-02, -5.57130426e-02,  9.14064199e-02,
        -1.14128888e-02,  1.37043297e-01],
       [-5.46909496e-02, -1.14931971e-01, -1.37827992e-02,
         3.52988616e-02,  4.73738164e-02, -1.70311034e-02,
         4.66488227e-02,  5.45501709e-04],
       [ 2.15902075e-01, -7.64941275e-02, -1.42735100e+00,
         3.98733020e-02,  3.65506038e-02, -2.09185332e-01,
        -7.47981071e-02,  1.40631199e-03],
       [ 5.09824753e-02, -9.08399522e-02,  1.21284127e-02,
        -5.23451641e-02, -9.91574526e-02,  3.09692770e-02,
        -4.39778604e-02,  2.02403367e-02],
       [-2.95685232e-03, -1.08291596e-01,  1.08269528e-02,
         5.99924326e-02,  6.10790253e-02,  2.99304612e-02,
         2.59107687e-02, -6.24357238e-02]], dtype=float32), array([[-0.01640466,  0.0250706 ,  0.2298055 , -0.24616392,  0.11454586,
        -0.01434946, -0.02582179, -0.03171336],
       [-0.15806663,  0.10450005, -0.0742714 , -0.11726975, -0.01447582,
        -0.03311225,  0.00366975, -0.03447901],
       [ 0.12898916,  0.04712423,  0.07286245, -0.1155621 ,  0.0017545 ,
         0.01885319,  0.1236327 , -0.03135961],
       [ 0.00273371,  0.01185413, -0.12779185, -0.13923708,  0.01008031,
         0.10026611,  0.26469046,  0.314048  ],
       [ 0.00520486,  0.01104314,  0.02281336,  0.14590396, -0.01829827,
        -0.00121284,  0.03487076, -0.05904349],
       [-0.0110303 , -0.08077252, -0.00803497,  0.0743899 ,  0.15235344,
         0.0201604 ,  0.00140951, -0.03169528],
       [-0.01472288, -0.00479572, -0.03975859, -0.02233921,  0.04339138,
         0.01349025,  0.00888905, -0.03250138],
       [ 0.0013186 ,  0.03723449, -0.02030665,  0.16565588,  0.04814844,
         0.05748284,  0.01524137,  0.0448309 ],
       [-0.0216642 ,  0.05673742, -0.00837889,  0.16109265,  0.01635757,
        -0.00647438,  0.02276647, -0.017922  ],
       [ 0.01851617, -0.04748037,  0.0124213 , -0.03625792,  0.08988935,
         0.01382854,  0.04276273, -0.03510501],
       [ 0.03491266, -0.04477507,  0.17033118, -0.02194342,  0.05952603,
         0.01581565,  0.06207959, -0.04095739],
       [ 0.0026761 , -0.03119664, -0.00203677, -0.5753358 ,  0.0489212 ,
         0.01647393, -0.06573971,  0.09610746],
       [ 0.38344166, -0.3383506 ,  0.00999045,  0.07093093, -0.34201634,
         0.05431442,  0.08254473,  0.02430499],
       [-0.05723231, -0.01559286,  0.03985616, -0.11295206, -0.02121723,
         0.29422328,  0.0376564 , -0.04375466],
       [ 0.02353678, -0.0517333 ,  0.16386026,  0.01688971, -0.03695262,
         0.06363398,  0.1135997 , -0.00507849],
       [-0.07273865, -0.16527803, -0.02589355, -0.16772635, -0.03979225,
        -0.00755751,  0.1225409 ,  0.02649874]], dtype=float32), array([[-1.4383793e-03],
       [ 2.7844906e-03],
       [-2.9643349e-02],
       [ 2.0581351e-01],
       [-3.9521813e-02],
       [ 1.1555891e-01],
       [-1.4499137e+00],
       [ 5.3399675e-02]], dtype=float32)]
>>> [(m1-m0)/m0 for (m1,m0) in zip(model1,model0)]
[array([[ 3.1000037e-02,  4.5698419e-01, -1.6908284e+00, -6.6792083e-01,
        -1.1627523e+00,  6.5312825e-02,  1.0816410e+00, -1.0722505e+00],
       [ 8.6416368e+00, -1.0849863e+00,  2.3599928e-03, -1.4190231e+02,
        -9.0403595e+00, -5.8054563e-02, -3.5560198e-02, -8.5807455e-01],
       [-9.0417573e-03,  1.9293247e-02, -3.8534952e-03, -7.6648816e-02,
         1.7588373e-02,  2.5250798e-01, -1.1088465e+00, -8.5578370e-01],
       [ 6.3630638e+00,  1.3816847e-02, -2.0822875e+00, -6.0072609e-05,
        -3.9991853e-01,  1.0299882e+01, -8.1534073e-02, -9.7203102e+00]],
      dtype=float32), array([[-2.5331875e-02, -1.4901191e+00, -1.2607111e+00, -1.9134070e+00,
        -7.3457783e-01, -8.1486903e-02,  3.2034828e-03, -1.8725437e-01],
       [-3.7303902e-03, -8.8926017e-02, -1.9337967e-02, -1.5451591e+01,
        -3.7968645e+00, -3.1658795e-02, -5.7811165e-01, -1.3489564e-01],
       [-1.3800163e+00,  1.0601988e-02,  4.3577687e-03, -6.3947082e-01,
         9.8008933e-03, -1.8026118e-04, -2.4967329e-01, -3.3486825e-01],
       [-8.8940972e-01, -2.4192898e-01,  8.8264858e-03,  1.8552338e-01,
        -3.0134210e-01, -1.7879577e+00, -3.2185640e-02,  6.5616856e+00],
       [-4.2780253e-01, -4.9729300e-01, -2.8334200e-01,  3.3205885e-01,
        -1.9810742e-01, -1.6179112e-01, -6.0624361e-01, -3.4793714e-04],
       [-1.9041567e+01, -3.6741066e-01, -1.8166045e+02,  2.6365840e-01,
         4.8636481e-01, -5.5484886e+00, -3.4776628e+00, -9.9696149e-04],
       [ 2.0366187e+00, -2.5171116e-01,  1.6382856e-02, -1.6925379e+00,
         2.3488102e+00, -1.6982548e-01, -5.0057483e-01, -1.3382635e-01],
       [ 5.3878948e-02, -2.9352579e-01,  1.1888563e-01,  3.4185576e-01,
         2.9179454e-01,  5.1322846e+00, -3.6597937e-01, -9.2813915e-01]],
      dtype=float32), array([[ 9.28165257e-01, -2.34467220e+00, -1.48182354e+01,
        -5.59680700e-01, -1.19324839e+00, -2.48046517e-02,
        -9.35449839e-01, -3.49225044e-01],
       [-3.39049006e+00,  2.43175197e+00, -1.06707549e+00,
        -5.09946421e-02,  8.56673997e-03, -3.25500101e-01,
        -2.49412563e-02, -3.63398438e+01],
       [ 9.38997841e+00, -5.64143121e-01,  2.00157213e+00,
        -5.32530785e+00, -2.76940893e-02, -2.16924131e-01,
         7.28956842e+00,  3.59708834e+00],
       [-2.72217183e-03, -5.10238819e-02, -2.14017129e+00,
         1.39369822e+01, -2.40481600e-01, -6.57714510e+00,
         4.18377647e+01, -1.98462341e+02],
       [-8.02093465e-03, -1.97284365e+00,  2.45278788e+00,
        -4.74912071e+00, -2.65787188e-02,  6.69052184e-04,
        -7.45592773e-01, -8.37085962e-01],
       [-5.56241944e-02, -1.25644374e+00, -2.92737521e-02,
         2.86784291e+00,  4.64041352e-01, -8.23967457e-01,
         4.07149345e-02, -3.39475304e-01],
       [-8.65472257e-02,  2.20304638e-01, -8.72880757e-01,
        -7.20486343e-01, -1.11519980e+00,  1.00059402e+00,
        -3.10390834e-02, -3.55400771e-01],
       [ 3.27692204e-03, -5.29188514e-01, -1.67757630e-01,
        -6.81356001e+00,  9.89741683e-01, -1.53318894e+00,
         1.31949437e+00, -5.76228321e-01],
       [-2.39749104e-02,  4.35603046e+00, -2.86160205e-02,
         1.02338161e+01, -8.43973532e-02, -6.11360651e-03,
         1.11716002e-01, -1.20978460e-01],
       [ 5.77850997e-01, -4.12843823e-01, -5.83047606e-02,
         6.26599863e-02,  4.12912035e+00, -3.71768743e-01,
         1.87527984e-01, -4.07191634e-01],
       [-9.05411422e-01, -9.90053415e-02, -3.43179631e+00,
        -6.15390599e-01, -1.50933516e+00, -1.17831714e-01,
        -6.50934041e-01, -5.18492579e-01],
       [ 1.19765306e+00, -2.60206431e-01,  7.48899281e-02,
         2.68065414e+01, -6.70036256e-01, -9.32620242e-02,
         1.07556581e+00, -7.77497330e+01],
       [-2.13165169e+01,  3.48829308e+01, -3.48949470e-02,
         2.84911275e-01, -1.15286140e+01, -3.47857881e+00,
        -6.53202653e-01, -2.11151928e-01],
       [-4.11794782e-01, -3.28187168e-01, -7.54428566e-01,
        -5.04567981e-01, -1.40421847e-02,  2.35900146e+02,
         1.01873004e+00,  1.76351562e+02],
       [ 5.58479929e+00, -1.88276720e+00,  6.96182060e+00,
         1.82597801e-01, -1.03669047e+00, -1.00590718e+00,
        -6.51418495e+00, -1.71257965e-02],
       [-1.19215417e+00, -6.10114276e-01, -2.54722804e-01,
        -4.30016428e-01,  5.05807638e-01,  1.12523241e-02,
         1.18614364e+00,  5.21850705e-01]], dtype=float32), array([[-1.4812329e-03],
       [ 2.9346864e-03],
       [ 7.1019411e-01],
       [-8.8411226e+00],
       [-3.3518746e-02],
       [-1.1143581e+00],
       [-1.0265794e+02],
       [-8.2728732e-01]], dtype=float32)]
>>> np.set_printoptions(suppress=True)
>>> [(m1-m0)/m0 for (m1,m0) in zip(model1,model0)]
[array([[   0.03100004,    0.4569842 ,   -1.6908284 ,   -0.6679208 ,
          -1.1627523 ,    0.06531283,    1.081641  ,   -1.0722505 ],
       [   8.641637  ,   -1.0849863 ,    0.00235999, -141.90231   ,
          -9.0403595 ,   -0.05805456,   -0.0355602 ,   -0.85807455],
       [  -0.00904176,    0.01929325,   -0.0038535 ,   -0.07664882,
           0.01758837,    0.25250798,   -1.1088465 ,   -0.8557837 ],
       [   6.363064  ,    0.01381685,   -2.0822875 ,   -0.00006007,
          -0.39991853,   10.299882  ,   -0.08153407,   -9.72031   ]],
      dtype=float32), array([[  -0.02533188,   -1.4901191 ,   -1.2607111 ,   -1.913407  ,
          -0.73457783,   -0.0814869 ,    0.00320348,   -0.18725437],
       [  -0.00373039,   -0.08892602,   -0.01933797,  -15.451591  ,
          -3.7968645 ,   -0.03165879,   -0.57811165,   -0.13489564],
       [  -1.3800163 ,    0.01060199,    0.00435777,   -0.6394708 ,
           0.00980089,   -0.00018026,   -0.24967329,   -0.33486825],
       [  -0.8894097 ,   -0.24192898,    0.00882649,    0.18552338,
          -0.3013421 ,   -1.7879577 ,   -0.03218564,    6.5616856 ],
       [  -0.42780253,   -0.497293  ,   -0.283342  ,    0.33205885,
          -0.19810742,   -0.16179112,   -0.6062436 ,   -0.00034794],
       [ -19.041567  ,   -0.36741066, -181.66045   ,    0.2636584 ,
           0.4863648 ,   -5.5484886 ,   -3.4776628 ,   -0.00099696],
       [   2.0366187 ,   -0.25171116,    0.01638286,   -1.6925379 ,
           2.3488102 ,   -0.16982548,   -0.5005748 ,   -0.13382635],
       [   0.05387895,   -0.2935258 ,    0.11888563,    0.34185576,
           0.29179454,    5.1322846 ,   -0.36597937,   -0.92813915]],
      dtype=float32), array([[   0.92816526,   -2.3446722 ,  -14.818235  ,   -0.5596807 ,
          -1.1932484 ,   -0.02480465,   -0.93544984,   -0.34922504],
       [  -3.39049   ,    2.431752  ,   -1.0670755 ,   -0.05099464,
           0.00856674,   -0.3255001 ,   -0.02494126,  -36.339844  ],
       [   9.389978  ,   -0.5641431 ,    2.0015721 ,   -5.325308  ,
          -0.02769409,   -0.21692413,    7.2895684 ,    3.5970883 ],
       [  -0.00272217,   -0.05102388,   -2.1401713 ,   13.936982  ,
          -0.2404816 ,   -6.577145  ,   41.837765  , -198.46234   ],
       [  -0.00802093,   -1.9728436 ,    2.4527879 ,   -4.7491207 ,
          -0.02657872,    0.00066905,   -0.7455928 ,   -0.83708596],
       [  -0.05562419,   -1.2564437 ,   -0.02927375,    2.867843  ,
           0.46404135,   -0.82396746,    0.04071493,   -0.3394753 ],
       [  -0.08654723,    0.22030464,   -0.87288076,   -0.72048634,
          -1.1151998 ,    1.000594  ,   -0.03103908,   -0.35540077],
       [   0.00327692,   -0.5291885 ,   -0.16775763,   -6.81356   ,
           0.9897417 ,   -1.5331889 ,    1.3194944 ,   -0.5762283 ],
       [  -0.02397491,    4.3560305 ,   -0.02861602,   10.233816  ,
          -0.08439735,   -0.00611361,    0.111716  ,   -0.12097846],
       [   0.577851  ,   -0.41284382,   -0.05830476,    0.06265999,
           4.1291203 ,   -0.37176874,    0.18752798,   -0.40719163],
       [  -0.9054114 ,   -0.09900534,   -3.4317963 ,   -0.6153906 ,
          -1.5093352 ,   -0.11783171,   -0.65093404,   -0.5184926 ],
       [   1.197653  ,   -0.26020643,    0.07488993,   26.806541  ,
          -0.67003626,   -0.09326202,    1.0755658 ,  -77.74973   ],
       [ -21.316517  ,   34.88293   ,   -0.03489495,    0.28491127,
         -11.528614  ,   -3.4785788 ,   -0.65320265,   -0.21115193],
       [  -0.41179478,   -0.32818717,   -0.75442857,   -0.504568  ,
          -0.01404218,  235.90015   ,    1.01873   ,  176.35156   ],
       [   5.5847993 ,   -1.8827672 ,    6.9618206 ,    0.1825978 ,
          -1.0366905 ,   -1.0059072 ,   -6.514185  ,   -0.0171258 ],
       [  -1.1921542 ,   -0.6101143 ,   -0.2547228 ,   -0.43001643,
           0.50580764,    0.01125232,    1.1861436 ,    0.5218507 ]],
      dtype=float32), array([[  -0.00148123],
       [   0.00293469],
       [   0.7101941 ],
       [  -8.841123  ],
       [  -0.03351875],
       [  -1.1143581 ],
       [-102.65794   ],
       [  -0.8272873 ]], dtype=float32)]
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 3s - loss: nan - val_loss: nan
<keras.callbacks.callbacks.History object at 0x000002E3B71D8EF0>
>>> from numpy import array
>>> from numpy import float32
>>> model.set_weights(weights1)
Traceback (most recent call last):
  File "<pyshell#850>", line 1, in <module>
    model.set_weights(weights1)
NameError: name 'weights1' is not defined
>>> model.set_weights(model1)
>>> y_pred=model.predict(x_train)
>>> model1[-2][-3]
array([ 0.08175029,  0.03191924, -0.01297344,  0.11090691,  1.4897462 ,
        0.2954705 ,  0.07462046, -0.04400277], dtype=float32)
>>> nans=y_pred.isnan()
Traceback (most recent call last):
  File "<pyshell#854>", line 1, in <module>
    nans=y_pred.isnan()
AttributeError: 'numpy.ndarray' object has no attribute 'isnan'
>>> np.isnan(y_pred).any()
False
>>> (y_pred==float('inf')).any()
False
>>> (y_pred==-float('inf')).any()
False
>>> tf.reduce_max(y_pred)
<tf.Tensor: id=39666, shape=(), dtype=float64, numpy=2.8208967699871663>
>>> tf.reduce_min(y_pred)
<tf.Tensor: id=39674, shape=(), dtype=float64, numpy=-3.0992487227519363>
>>> [l0_1(w) for w in model1]
[<tf.Tensor: id=39682, shape=(), dtype=float64, numpy=31.256017684936523>, <tf.Tensor: id=39689, shape=(), dtype=float64, numpy=62.73912811279297>, <tf.Tensor: id=39696, shape=(), dtype=float64, numpy=124.81359100341797>, <tf.Tensor: id=39703, shape=(), dtype=float64, numpy=7.87386417388916>]
>>> model.evaluate(x_train,y_train)
  32/8000 [..............................] - ETA: 1s  64/8000 [..............................] - ETA: 11s  96/8000 [..............................] - ETA: 16s 128/8000 [..............................] - ETA: 18s 160/8000 [..............................] - ETA: 20s 192/8000 [..............................] - ETA: 23s 224/8000 [..............................] - ETA: 24s 256/8000 [..............................] - ETA: 25s 288/8000 [>.............................] - ETA: 26s 320/8000 [>.............................] - ETA: 26sTraceback (most recent call last):
  File "<pyshell#861>", line 1, in <module>
    model.evaluate(x_train,y_train)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training.py", line 1361, in evaluate
    callbacks=callbacks)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\training_arrays.py", line 468, in test_loop
    progbar.update(batch_end)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\utils\generic_utils.py", line 385, in update
    sys.stdout.write('\b' * prev_total_width)
KeyboardInterrupt
>>> model.evaluate(x_train,y_train,verbose=0)
227.19893804073334
>>> model.loss
<function complex_absolute_error at 0x000002E3B5986488>
>>> model.loss()
Traceback (most recent call last):
  File "<pyshell#864>", line 1, in <module>
    model.loss()
TypeError: complex_absolute_error() missing 2 required positional arguments: 'y_true' and 'y_pred'
>>> g=sgd.get_gradients(model.loss(y_train,model.predict(x_train)))
Traceback (most recent call last):
  File "<pyshell#865>", line 1, in <module>
    g=sgd.get_gradients(model.loss(y_train,model.predict(x_train)))
TypeError: get_gradients() missing 1 required positional argument: 'params'
>>> g=sgd.get_gradients(model.loss(y_train,model.predict(x_train)),model.weights)
Traceback (most recent call last):
  File "<pyshell#866>", line 1, in <module>
    g=sgd.get_gradients(model.loss(y_train,model.predict(x_train)),model.weights)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\optimizers.py", line 93, in get_gradients
    raise ValueError('An operation has `None` for gradient. '
ValueError: An operation has `None` for gradient. Please make sure that all of your ops have a gradient defined (i.e. are differentiable). Common ops without gradient: K.argmax, K.round, K.eval.
>>> xs = tf.constant([random()*10 for _ in range(10)])
>>> xs
<tf.Tensor: id=42533, shape=(10,), dtype=float32, numpy=
array([4.009541 , 2.394453 , 2.6177227, 8.856096 , 2.8910613, 4.048342 ,
       8.443878 , 3.2273614, 1.4226174, 1.8930898], dtype=float32)>
>>> ys = tf.math.log(xs)
>>> tf.gradients(ys,xs)
Traceback (most recent call last):
  File "<pyshell#870>", line 1, in <module>
    tf.gradients(ys,xs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gradients_impl.py", line 274, in gradients_v2
    unconnected_gradients)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gradients_util.py", line 504, in _GradientsHelper
    raise RuntimeError("tf.gradients is not supported when eager execution "
RuntimeError: tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead.
>>> tf.GradientTape(ys,xs)
<tensorflow.python.eager.backprop.GradientTape object at 0x000002E3BA0F08D0>
>>> ys = tf.round(xs)
>>> tf.GradientTape(ys,xs)
<tensorflow.python.eager.backprop.GradientTape object at 0x000002E3BA0F0128>
>>> tf.GradientTape(ys,xs).eval()
Traceback (most recent call last):
  File "<pyshell#874>", line 1, in <module>
    tf.GradientTape(ys,xs).eval()
AttributeError: 'GradientTape' object has no attribute 'eval'
>>> ys = K.round(xs)
>>> tf.GradientTape(ys,xs)
<tensorflow.python.eager.backprop.GradientTape object at 0x000002E3BA0F0A20>
>>> xs = tf.constant([[random()*10] for _ in range(10)])
>>> xs.shape
TensorShape([10, 1])
>>> ys = tf.complex(xs,tf.zeros_like(xs))
>>> tf.GradientTape(ys,xs)
<tensorflow.python.eager.backprop.GradientTape object at 0x000002E3BA0F0978>
>>> with tf.GradientTape() as g:
	g.watch(x)
	g.watch(xs)
	ys = tf.complex(xs,tf.zeros_like(xs))
dy_dx = g.gradient(ys, xs)
SyntaxError: invalid syntax
>>> xs = tf.constant([[random()*10] for _ in range(10)])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.complex(xs,tf.zeros_like(xs))

	
>>> with tf.GradientTape() as g:
	g.watch(x)
	g.watch(xs)
	ys = tf.complex(xs,tf.zeros_like(xs))
dy_dx = g.gradient(ys, xs)
SyntaxError: invalid syntax
>>> dy_dx = g.gradient(ys, xs)
WARNING:tensorflow:The dtype of the target tensor must be floating (e.g. tf.float32) when calling GradientTape.gradient, got tf.complex64
>>> xs = tf.constant([[random()*20-10] for _ in range(10)])
>>> xs
<tf.Tensor: id=42557, shape=(10, 1), dtype=float32, numpy=
array([[-0.1624674 ],
       [-8.4788065 ],
       [-0.13658339],
       [ 6.700089  ],
       [ 1.5462565 ],
       [ 4.5136843 ],
       [ 8.169832  ],
       [-2.2277815 ],
       [-4.5521865 ],
       [ 3.1457458 ]], dtype=float32)>
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.real(tf.sqrt(tf.complex(xs,tf.zeros_like(xs))))

	
Traceback (most recent call last):
  File "<pyshell#896>", line 3, in <module>
    ys = tf.real(tf.sqrt(tf.complex(xs,tf.zeros_like(xs))))
AttributeError: module 'tensorflow' has no attribute 'real'
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.math.real(tf.sqrt(tf.complex(xs,tf.zeros_like(xs))))

	
>>> with tf.GradientTape() as g:
	g.watch(x)
	g.watch(xs)
	ys = tf.complex(xs,tf.zeros_like(xs))
dy_dx = g.gradient(ys, xs)
SyntaxError: invalid syntax
>>> dy_dx = g.gradient(ys, xs)
>>> dy_dx
<tf.Tensor: id=42575, shape=(10, 1), dtype=float32, numpy=
array([[ 0.        ],
       [ 0.        ],
       [ 0.        ],
       [ 0.19316557],
       [ 0.4020955 ],
       [ 0.23534468],
       [ 0.17492965],
       [ 0.        ],
       [-0.        ],
       [ 0.2819085 ]], dtype=float32)>
>>> ys
<tf.Tensor: id=42562, shape=(10, 1), dtype=float32, numpy=
array([[0.       ],
       [0.       ],
       [0.       ],
       [2.588453 ],
       [1.2434857],
       [2.1245434],
       [2.8582919],
       [0.       ],
       [0.       ],
       [1.773625 ]], dtype=float32)>
>>> xs
<tf.Tensor: id=42557, shape=(10, 1), dtype=float32, numpy=
array([[-0.1624674 ],
       [-8.4788065 ],
       [-0.13658339],
       [ 6.700089  ],
       [ 1.5462565 ],
       [ 4.5136843 ],
       [ 8.169832  ],
       [-2.2277815 ],
       [-4.5521865 ],
       [ 3.1457458 ]], dtype=float32)>
>>> tf.math.real(tf.sqrt(tf.complex(xs,tf.zeros_like(xs))))
<tf.Tensor: id=42585, shape=(10, 1), dtype=float32, numpy=
array([[0.       ],
       [0.       ],
       [0.       ],
       [2.588453 ],
       [1.2434857],
       [2.1245434],
       [2.8582919],
       [0.       ],
       [0.       ],
       [1.773625 ]], dtype=float32)>
>>> tf.sqrt(tf.complex(xs,tf.zeros_like(xs)))
<tf.Tensor: id=42589, shape=(10, 1), dtype=complex64, numpy=
array([[0.       +0.40307245j],
       [0.       +2.911839j  ],
       [0.       +0.3695719j ],
       [2.588453 +0.j        ],
       [1.2434857+0.j        ],
       [2.1245434+0.j        ],
       [2.8582919+0.j        ],
       [0.       +1.4925755j ],
       [0.       +2.1335855j ],
       [1.773625 +0.j        ]], dtype=complex64)>
>>> tf.math.real(tf.pow(tf.complex(xs,tf.zeros_like(xs)),0.3))
<tf.Tensor: id=42595, shape=(10, 1), dtype=float32, numpy=
array([[0.3407599 ],
       [1.1161411 ],
       [0.32347298],
       [1.7693925 ],
       [1.1396841 ],
       [1.5716628 ],
       [1.8778632 ],
       [0.7474471 ],
       [0.9261571 ],
       [1.4103181 ]], dtype=float32)>
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.math.real(tf.pow(tf.complex(xs,tf.zeros_like(xs)),0.3))

	
>>> dy_dx = g.gradient(ys, xs)
>>> dy_dx
<tf.Tensor: id=42640, shape=(10, 1), dtype=float32, numpy=
array([[-0.62922144],
       [-0.03949168],
       [-0.7104955 ],
       [ 0.07922548],
       [ 0.22111805],
       [ 0.10445987],
       [ 0.068956  ],
       [-0.10065357],
       [-0.06103597],
       [ 0.13449767]], dtype=float32)>
>>> xs
<tf.Tensor: id=42557, shape=(10, 1), dtype=float32, numpy=
array([[-0.1624674 ],
       [-8.4788065 ],
       [-0.13658339],
       [ 6.700089  ],
       [ 1.5462565 ],
       [ 4.5136843 ],
       [ 8.169832  ],
       [-2.2277815 ],
       [-4.5521865 ],
       [ 3.1457458 ]], dtype=float32)>
>>> us
Traceback (most recent call last):
  File "<pyshell#912>", line 1, in <module>
    us
NameError: name 'us' is not defined
>>> ys
<tf.Tensor: id=42601, shape=(10, 1), dtype=float32, numpy=
array([[0.3407599 ],
       [1.1161411 ],
       [0.32347298],
       [1.7693925 ],
       [1.1396841 ],
       [1.5716628 ],
       [1.8778632 ],
       [0.7474471 ],
       [0.9261571 ],
       [1.4103181 ]], dtype=float32)>
>>> xs.shape
TensorShape([10, 1])
>>> with tf.GradientTape() as g:
	
KeyboardInterrupt
>>> len(model1)
4
>>> model1[0]/shape
Traceback (most recent call last):
  File "<pyshell#917>", line 1, in <module>
    model1[0]/shape
NameError: name 'shape' is not defined
>>> model1[0].shape
(4, 8)
>>> def call(self, inputs, kernel):
        if len(inputs.shape)==2:
            inputs = tf.cast(tf.complex(inputs,tf.zeros_like(inputs)),tf.complex128)
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.cast(tf.complex(input_real,input_imag),tf.complex128)
        if use_log:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)
        inputs = complex_clip(inputs)
        #USE RECTIFIED LOG ON THE WEIGHTS MATRIX SO THE SMALL ONES WILL ACTUALLY BE ZERO
        outputs = tf.matmul(inputs,tf.cast(tf.complex(self.kernel,tf.zeros_like(kernel)),tf.complex128))
        if use_log:
            outputs = tf.exp(outputs)
        outputs = complex_clip(outputs)
        output_real = tf.math.real(outputs)
        output_imag = tf.math.imag(outputs)
        outputs = tf.stack((output_real,output_imag),axis=2)
        return outputs

>>> inputs = tf.constant([[random()*20-10] for _ in range(20)])
>>> inputs.shape
TensorShape([20, 1])
>>> def call(inputs, kernel):
        if len(inputs.shape)==2:
            inputs = tf.cast(tf.complex(inputs,tf.zeros_like(inputs)),tf.complex128)
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.cast(tf.complex(input_real,input_imag),tf.complex128)
        if use_log:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)
        inputs = complex_clip(inputs)
        #USE RECTIFIED LOG ON THE WEIGHTS MATRIX SO THE SMALL ONES WILL ACTUALLY BE ZERO
        outputs = tf.matmul(inputs,tf.cast(tf.complex(self.kernel,tf.zeros_like(kernel)),tf.complex128))
        if use_log:
            outputs = tf.exp(outputs)
        outputs = complex_clip(outputs)
        output_real = tf.math.real(outputs)
        output_imag = tf.math.imag(outputs)
        outputs = tf.stack((output_real,output_imag),axis=2)
        return outputs

>>> with tf.GradientTape() as g:
	g.watch(inputs)
	inputs = call(inputs,model1[0],True)

	
Traceback (most recent call last):
  File "<pyshell#926>", line 3, in <module>
    inputs = call(inputs,model1[0],True)
TypeError: call() takes 2 positional arguments but 3 were given
>>> def call(inputs, kernel, use_log):
        if len(inputs.shape)==2:
            inputs = tf.cast(tf.complex(inputs,tf.zeros_like(inputs)),tf.complex128)
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.cast(tf.complex(input_real,input_imag),tf.complex128)
        if use_log:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)
        inputs = complex_clip(inputs)
        #USE RECTIFIED LOG ON THE WEIGHTS MATRIX SO THE SMALL ONES WILL ACTUALLY BE ZERO
        outputs = tf.matmul(inputs,tf.cast(tf.complex(self.kernel,tf.zeros_like(kernel)),tf.complex128))
        if use_log:
            outputs = tf.exp(outputs)
        outputs = complex_clip(outputs)
        output_real = tf.math.real(outputs)
        output_imag = tf.math.imag(outputs)
        outputs = tf.stack((output_real,output_imag),axis=2)
        return outputs

>>> with tf.GradientTape() as g:
	g.watch(inputs)
	inputs1 = call(inputs,model1[0],True)
	inputs2 = call(inputs,model1[1],False)
	inputs3 = call(inputs,model1[2],True)
	outputs = call(inputs,model1[3],False)

	
Traceback (most recent call last):
  File "<pyshell#931>", line 3, in <module>
    inputs1 = call(inputs,model1[0],True)
  File "<pyshell#928>", line 13, in call
    outputs = tf.matmul(inputs,tf.cast(tf.complex(self.kernel,tf.zeros_like(kernel)),tf.complex128))
NameError: name 'self' is not defined
>>> def call(inputs, kernel, use_log):
        if len(inputs.shape)==2:
            inputs = tf.cast(tf.complex(inputs,tf.zeros_like(inputs)),tf.complex128)
        else:
            input_real = inputs[:,:,0]
            input_imag = inputs[:,:,1]
            inputs = tf.cast(tf.complex(input_real,input_imag),tf.complex128)
        if use_log:
            input_log = tf.math.log(inputs)
            inputs = tf.concat([inputs,input_log],axis=1)
        inputs = complex_clip(inputs)
        #USE RECTIFIED LOG ON THE WEIGHTS MATRIX SO THE SMALL ONES WILL ACTUALLY BE ZERO
        outputs = tf.matmul(inputs,tf.cast(tf.complex(kernel,tf.zeros_like(kernel)),tf.complex128))
        if use_log:
            outputs = tf.exp(outputs)
        outputs = complex_clip(outputs)
        output_real = tf.math.real(outputs)
        output_imag = tf.math.imag(outputs)
        outputs = tf.stack((output_real,output_imag),axis=2)
        return outputs

>>> with tf.GradientTape() as g:
	g.watch(inputs)
	inputs1 = call(inputs,model1[0],True)
	inputs2 = call(inputs,model1[1],False)
	inputs3 = call(inputs,model1[2],True)
	outputs = call(inputs,model1[3],False)

	
Traceback (most recent call last):
  File "<pyshell#935>", line 3, in <module>
    inputs1 = call(inputs,model1[0],True)
  File "<pyshell#933>", line 13, in call
    outputs = tf.matmul(inputs,tf.cast(tf.complex(kernel,tf.zeros_like(kernel)),tf.complex128))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 2647, in matmul
    a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 6284, in mat_mul
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: Matrix size-incompatible: In[0]: [20,2], In[1]: [4,8] [Op:MatMul] name: MatMul/
>>> len(inputs.shape)
2
>>> with tf.GradientTape() as g:
	g.watch(inputs)
	inputs1 = call(inputs,model1[0],True)

	
Traceback (most recent call last):
  File "<pyshell#938>", line 3, in <module>
    inputs1 = call(inputs,model1[0],True)
  File "<pyshell#933>", line 13, in call
    outputs = tf.matmul(inputs,tf.cast(tf.complex(kernel,tf.zeros_like(kernel)),tf.complex128))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 2647, in matmul
    a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 6284, in mat_mul
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: Matrix size-incompatible: In[0]: [20,2], In[1]: [4,8] [Op:MatMul] name: MatMul/
>>> call(inputs,model0[0],True)
Traceback (most recent call last):
  File "<pyshell#939>", line 1, in <module>
    call(inputs,model0[0],True)
  File "<pyshell#933>", line 13, in call
    outputs = tf.matmul(inputs,tf.cast(tf.complex(kernel,tf.zeros_like(kernel)),tf.complex128))
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py", line 2647, in matmul
    a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 6284, in mat_mul
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: Matrix size-incompatible: In[0]: [20,2], In[1]: [4,8] [Op:MatMul] name: MatMul/
>>> if len(inputs.shape)==2:
	inputs = tf.cast(tf.complex(inputs,tf.zeros_like(inputs)),tf.complex128)
else:
	input_real = inputs[:,:,0]
	input_imag = inputs[:,:,1]
	inputs = tf.cast(tf.complex(input_real,input_imag),tf.complex128)

	
>>> inputs.shape
TensorShape([20, 1])
>>> if use_log:
	input_log = tf.math.log(inputs)
	inputs = tf.concat([inputs,input_log],axis=1)

	
Traceback (most recent call last):
  File "<pyshell#944>", line 1, in <module>
    if use_log:
NameError: name 'use_log' is not defined
>>> if True:
	input_log = tf.math.log(inputs)
	inputs = tf.concat([inputs,input_log],axis=1)

	
>>> inputs.shape
TensorShape([20, 2])
>>> inputs_log.shape
Traceback (most recent call last):
  File "<pyshell#948>", line 1, in <module>
    inputs_log.shape
NameError: name 'inputs_log' is not defined
>>> input_log.shape
TensorShape([20, 1])
>>> inputs.shape
TensorShape([20, 2])
>>> xs = tf.constant([[random()*20-10,random()*20-10] for _ in range(10)])
>>> xs = tf.constant([[random()*20-10,random()*20-10] for _ in range(10)])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	inputs = call(xs,model1[0],True)
	inputs = call(inputs,model1[1],False)
	inputs = call(inputs,model1[2],True)
	output = call(inputs,model1[3],False)

	
>>> dy_dx = g.gradient(output, xs)
>>> dy_dx.shape
TensorShape([10, 2])
>>> dy_dx
<tf.Tensor: id=43228, shape=(10, 2), dtype=float32, numpy=
array([[ 8.8719009e+12,  2.0260258e+12],
       [ 8.7440045e+08,  6.5028905e+09],
       [ 1.9600690e-39, -7.1526702e-39],
       [-1.3115070e+12,  4.7966076e+12],
       [           nan,            nan],
       [-1.1216975e+00, -2.8769720e+02],
       [           nan,            nan],
       [ 8.9369440e+02, -7.0718622e+02],
       [-2.0431151e+00, -2.8081865e+00],
       [ 5.4426432e-41, -7.9719870e-41]], dtype=float32)>
>>> xs
<tf.Tensor: id=42738, shape=(10, 2), dtype=float32, numpy=
array([[ 0.689764  ,  8.56142   ],
       [ 3.776528  ,  3.7555535 ],
       [ 5.306349  , -1.7024244 ],
       [ 8.146296  ,  5.258552  ],
       [-6.632179  , -7.429182  ],
       [-9.789708  ,  0.19843364],
       [-7.1124477 , -6.5524497 ],
       [ 1.0240859 , -2.4814708 ],
       [-0.36167714, -0.57339627],
       [ 1.8586948 , -3.1155832 ]], dtype=float32)>
>>> outputs
Traceback (most recent call last):
  File "<pyshell#959>", line 1, in <module>
    outputs
NameError: name 'outputs' is not defined
>>> output
<tf.Tensor: id=42900, shape=(10, 1, 2), dtype=float64, numpy=
array([[[ 3.79493441e+11,  3.18631578e+11]],

       [[ 5.55114147e+08,  4.65771827e+08]],

       [[ 1.01693022e+13, -1.01693022e+13]],

       [[ 4.62940293e+11,  3.88677014e+11]],

       [[ 1.01693022e+13,  1.01693022e+13]],

       [[ 9.86401266e+00,  1.16734356e+01]],

       [[ 1.01693022e+13,  1.01693022e+13]],

       [[ 8.03764575e+01,  3.34292192e+01]],

       [[ 2.25681202e-01,  6.23805820e-02]],

       [[ 1.01693022e+13,  1.01693022e+13]]])>
>>> output.shape
TensorShape([10, 1, 2])
>>> output[:,0,:]
<tf.Tensor: id=43238, shape=(10, 2), dtype=float64, numpy=
array([[ 3.79493441e+11,  3.18631578e+11],
       [ 5.55114147e+08,  4.65771827e+08],
       [ 1.01693022e+13, -1.01693022e+13],
       [ 4.62940293e+11,  3.88677014e+11],
       [ 1.01693022e+13,  1.01693022e+13],
       [ 9.86401266e+00,  1.16734356e+01],
       [ 1.01693022e+13,  1.01693022e+13],
       [ 8.03764575e+01,  3.34292192e+01],
       [ 2.25681202e-01,  6.23805820e-02],
       [ 1.01693022e+13,  1.01693022e+13]])>
>>> xs = tf.constant([-100,-2,10,1000])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.clip_by_value(xs,-50,50)

	
WARNING:tensorflow:The dtype of the watched tensor must be floating (e.g. tf.float32), got tf.int32
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.clip_by_value(xs,-50.,50.)

	
WARNING:tensorflow:The dtype of the watched tensor must be floating (e.g. tf.float32), got tf.int32
Traceback (most recent call last):
  File "<pyshell#967>", line 3, in <module>
    ys = tf.clip_by_value(xs,-50.,50.)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\clip_ops.py", line 72, in clip_by_value
    t_min = math_ops.minimum(values, clip_value_max)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 6705, in minimum
    name, _ctx._post_execution_callbacks, x, y)
TypeError: Cannot convert provided value to EagerTensor. Provided value: 50.0 Requested dtype: int32
>>> xs = tf.constant([-100.,-2.,10.,1000.])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.clip_by_value(xs,-50.,50.)

	
>>> dy_dx = g.gradient(output, xs)
>>> dy_dx
>>> dy_dx
>>> xs = tf.constant([-100.,-2.,10.,1000.])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.clip_by_value(xs,-5000.,5000.)

	
>>> dy_dx = g.gradient(output, xs)
>>> xs = tf.constant([-100.,-2.,10.,1000.])
>>> dy_dx
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.add(xs,-5000.)

	
>>> xs = tf.constant([-100.,-2.,10.,1000.])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.clip_by_value(xs,-5000.,5000.)

	
>>> dy_dx = g.gradient(ys, xs)
>>> dy_dx
<tf.Tensor: id=43271, shape=(4,), dtype=float32, numpy=array([1., 1., 1., 1.], dtype=float32)>
>>> xs = tf.constant([-100.,-2.,10.,1000.])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.clip_by_value(xs,-50.,50.)

	
>>> dy_dx = g.gradient(ys, xs)
>>> dy_dx
<tf.Tensor: id=43286, shape=(4,), dtype=float32, numpy=array([0., 1., 1., 0.], dtype=float32)>
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.round(xs)

	
>>> dy_dx = g.gradient(ys, xs)
>>> dy_dx
>>> g.gradient(ys, xs)
Traceback (most recent call last):
  File "<pyshell#997>", line 1, in <module>
    g.gradient(ys, xs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\backprop.py", line 953, in gradient
    raise RuntimeError("GradientTape.gradient can only be called once on "
RuntimeError: GradientTape.gradient can only be called once on non-persistent tapes.
>>> xs = tf.constant([-100.,-2.,10.,1000.])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.round(xs)

	
>>> g.gradient(ys, xs)
>>> g.gradient(ys, xs)
Traceback (most recent call last):
  File "<pyshell#1002>", line 1, in <module>
    g.gradient(ys, xs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\backprop.py", line 953, in gradient
    raise RuntimeError("GradientTape.gradient can only be called once on "
RuntimeError: GradientTape.gradient can only be called once on non-persistent tapes.
>>> xs = tf.constant([-100.,-2.,10.,1000.])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.div(xs,2)

	
Traceback (most recent call last):
  File "<pyshell#1005>", line 3, in <module>
    ys = tf.div(xs,2)
AttributeError: module 'tensorflow' has no attribute 'div'
>>> with tf.GradientTape() as g:
	g.watch(xs)
	ys = tf.divide(xs,2)

	
>>> g.gradient(ys, xs)
<tf.Tensor: id=43309, shape=(4,), dtype=float32, numpy=array([0.5, 0.5, 0.5, 0.5], dtype=float32)>
>>> g.gradient(ys, xs)
Traceback (most recent call last):
  File "<pyshell#1009>", line 1, in <module>
    g.gradient(ys, xs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\backprop.py", line 953, in gradient
    raise RuntimeError("GradientTape.gradient can only be called once on "
RuntimeError: GradientTape.gradient can only be called once on non-persistent tapes.
>>> xs = tf.constant([[random()*20-10,random()*20-10] for _ in range(10)])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	inputs = call(xs,model1[0],True)
	inputs = call(inputs,model1[1],False)
	inputs = call(inputs,model1[2],True)
	output = call(inputs,model1[3],False)

	
>>> dy_dx = g.gradient(output, xs)
>>> dy_dx
<tf.Tensor: id=43807, shape=(10, 2), dtype=float32, numpy=
array([[-3.3373657e+01,  5.1372343e-01],
       [ 0.0000000e+00,  0.0000000e+00],
       [ 3.7830564e+09,  3.5525420e+09],
       [-1.4457523e+02, -6.0638852e+00],
       [ 0.0000000e+00,  0.0000000e+00],
       [ 3.8035684e+00,  1.7288623e+00],
       [ 0.0000000e+00,  0.0000000e+00],
       [ 1.2062958e+10,  2.5141027e+10],
       [ 1.6762064e+04, -1.7993506e+04],
       [ 1.1922313e+07,  4.3295760e+06]], dtype=float32)>
>>> xs = tf.constant([[random()*20-10,random()*20-10] for _ in range(10)])
>>> with tf.GradientTape() as g:
	g.watch(xs)
	inputs = call(xs,model1[0],True)
	inputs = call(inputs,model1[1],False)
	inputs = call(inputs,model1[2],True)
	output = call(inputs,model1[3],False)

	
>>> dy_dx = g.gradient(output, xs)
>>> dy_dx
<tf.Tensor: id=44302, shape=(10, 2), dtype=float32, numpy=
array([[ 0.0000000e+00,  0.0000000e+00],
       [           nan,            nan],
       [ 5.6207970e+10, -1.4475854e+11],
       [           nan,            nan],
       [ 0.0000000e+00,  0.0000000e+00],
       [-7.3262214e+08,  7.5208928e+08],
       [ 0.0000000e+00,  0.0000000e+00],
       [           nan,            nan],
       [ 9.5655562e+05,  5.5996144e+05],
       [           nan,            nan]], dtype=float32)>
>>> xs
<tf.Tensor: id=43812, shape=(10, 2), dtype=float32, numpy=
array([[-5.4963684,  5.143496 ],
       [ 4.0224204, -9.932158 ],
       [ 7.3293076,  6.7533965],
       [-9.666206 , -7.253449 ],
       [-4.8410907,  7.09622  ],
       [-3.775501 ,  2.0934844],
       [ 9.712578 , -2.1351585],
       [ 5.8121686, -3.4663675],
       [ 7.5758953,  9.589514 ],
       [ 9.792761 , -2.9374094]], dtype=float32)>
>>> data = [[(random()*2-1)*m,(random()*2-1)*m] for _ in range(10000)]
>>> x_train = np.array(data[:Nt])
>>> with tf.GradientTape() as g:
	g.watch(x_train)
	inputs = call(x_train,model1[0],True)
	inputs = call(inputs,model1[1],False)
	inputs = call(inputs,model1[2],True)
	output = call(inputs,model1[3],False)

	
Traceback (most recent call last):
  File "<pyshell#1024>", line 2, in <module>
    g.watch(x_train)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\backprop.py", line 839, in watch
    if not t.dtype.is_floating:
AttributeError: 'numpy.dtype' object has no attribute 'is_floating'
>>> x_train.shape
(8000, 2)
>>> xs.shape
TensorShape([10, 2])
>>> inputs = tf.convert_to_tensor(x_train)
>>> inputs.shape
TensorShape([8000, 2])
>>> with tf.GradientTape() as g:
	g.watch(x_train)
	inputs = call(x_train,model1[0],True)
	inputs = call(inputs,model1[1],False)
	inputs = call(inputs,model1[2],True)
	output = call(inputs,model1[3],False)

	
Traceback (most recent call last):
  File "<pyshell#1030>", line 2, in <module>
    g.watch(x_train)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\backprop.py", line 839, in watch
    if not t.dtype.is_floating:
AttributeError: 'numpy.dtype' object has no attribute 'is_floating'
>>> xs[:10]
<tf.Tensor: id=44312, shape=(10, 2), dtype=float32, numpy=
array([[-5.4963684,  5.143496 ],
       [ 4.0224204, -9.932158 ],
       [ 7.3293076,  6.7533965],
       [-9.666206 , -7.253449 ],
       [-4.8410907,  7.09622  ],
       [-3.775501 ,  2.0934844],
       [ 9.712578 , -2.1351585],
       [ 5.8121686, -3.4663675],
       [ 7.5758953,  9.589514 ],
       [ 9.792761 , -2.9374094]], dtype=float32)>
>>> x_train[:10]
array([[-0.41031594, -0.66657667],
       [ 0.33589886,  0.03264617],
       [ 0.7119622 , -0.08091059],
       [ 0.80924749, -0.53375929],
       [ 0.24650721,  0.31043562],
       [-0.45484573,  0.47460738],
       [-0.69122503,  0.43677941],
       [ 0.18567934,  0.73858574],
       [ 0.43108302, -0.69935024],
       [-0.89793039, -0.75307822]])
>>> inputs = tf.convert_to_tensor(x_train,float32)
>>> with tf.GradientTape() as g:
	g.watch(inputs)
	inputs1 = call(x_train,model1[0],True)
	inputs1 = call(inputs1,model1[1],False)
	inputs1 = call(inputs1,model1[2],True)
	output = call(inputs1,model1[3],False)

	
>>> dy_dx = g.gradient(output, inputs)
>>> dy_dx.shape
Traceback (most recent call last):
  File "<pyshell#1037>", line 1, in <module>
    dy_dx.shape
AttributeError: 'NoneType' object has no attribute 'shape'
>>> inputs.shape
TensorShape([8000, 2])
>>> inptuts[:0,:0]
Traceback (most recent call last):
  File "<pyshell#1039>", line 1, in <module>
    inptuts[:0,:0]
NameError: name 'inptuts' is not defined
>>> inputs[:0,:0]
<tf.Tensor: id=44481, shape=(0, 0), dtype=float32, numpy=array([], shape=(0, 0), dtype=float32)>
>>> output[:0,:0]
<tf.Tensor: id=44486, shape=(0, 0, 2), dtype=float64, numpy=array([], shape=(0, 0, 2), dtype=float64)>
>>> inputs[:0]
<tf.Tensor: id=44491, shape=(0, 2), dtype=float32, numpy=array([], shape=(0, 2), dtype=float32)>
>>> output[:0]
<tf.Tensor: id=44496, shape=(0, 1, 2), dtype=float64, numpy=array([], shape=(0, 1, 2), dtype=float64)>
>>> np.isnan(output).any()
False
>>> np.max(output)
2.7962550582401335
>>> np.min(output)
-2.0248874265196357
>>> inputs.shape
TensorShape([8000, 2])
>>> inputs[:10]
<tf.Tensor: id=44501, shape=(10, 2), dtype=float32, numpy=
array([[-0.41031593, -0.6665767 ],
       [ 0.33589885,  0.03264617],
       [ 0.7119622 , -0.08091059],
       [ 0.8092475 , -0.5337593 ],
       [ 0.24650721,  0.31043562],
       [-0.45484573,  0.47460738],
       [-0.69122505,  0.4367794 ],
       [ 0.18567933,  0.73858577],
       [ 0.43108302, -0.69935024],
       [-0.8979304 , -0.7530782 ]], dtype=float32)>
>>> with tf.GradientTape(persistent=True) as g:
	g.watch(inputs)
	inputs1 = call(inputs,model1[0],True)
	inputs1 = call(inputs1,model1[1],False)
	inputs1 = call(inputs1,model1[2],True)
	output = call(inputs1,model1[3],False)

	
>>> dy_dx = g.gradient(output, inputs)
>>> dy_dx.shape
TensorShape([8000, 2])
>>> dy_dx[:10]
<tf.Tensor: id=44999, shape=(10, 2), dtype=float32, numpy=
array([[-0.85304594, -1.9799964 ],
       [-5.965289  , -0.9452838 ],
       [-8.679141  ,  2.9900262 ],
       [ 1.6979909 , -2.1330183 ],
       [-2.6650462 , -1.9369378 ],
       [-1.4773425 ,  0.12211902],
       [-1.2340525 , -0.12745136],
       [ 1.0794153 ,  2.9874659 ],
       [-6.230548  ,  4.588517  ],
       [-2.3614416 , -2.5839376 ]], dtype=float32)>
>>> np.isnan(dy_dx).any()
False
>>> target=[[x*1.1**(x*y)+np.sqrt(x*x+y*y)] for [x,y] in data]
>>> y_train = np.array(target[:Nt])
>>> with tf.GradientTape(persistent=True) as g:
	g.watch(inputs)
	inputs1 = call(inputs,model1[0],True)
	inputs1 = call(inputs1,model1[1],False)
	inputs1 = call(inputs1,model1[2],True)
	output = call(inputs1,model1[3],False)
	loss = complex_mean_absolute_error(y_train, output)

	
(8000, 1) (8000, 1, 2)
>>> dy_dx = g.gradient(loss, inputs)
>>> dy_dx.shape
TensorShape([8000, 2])
>>> dy_dx[:10]
<tf.Tensor: id=45514, shape=(10, 2), dtype=float32, numpy=
array([[-0.00005332, -0.00012375],
       [-0.0001588 , -0.0003248 ],
       [-0.00033803,  0.00048218],
       [-0.00010612,  0.00013331],
       [-0.00001616, -0.00014285],
       [ 0.00009233, -0.00000763],
       [ 0.00000508, -0.00007253],
       [-0.00005369, -0.0000314 ],
       [-0.00020117,  0.00025218],
       [-0.00014759, -0.0001615 ]], dtype=float32)>
>>> weight_matrix = tf.constant(model0[0])
>>> with tf.GradientTape(persistent=True) as g:
	g.watch(weight_matrix)
	reg = l0_1(weight_matrix)

	
>>> dy_dx = g.gradient(reg, weight_matrix)
>>> dy_dx
<tf.Tensor: id=45555, shape=(4, 8), dtype=float32, numpy=
array([[ -0.09162604,  -0.07311635,  -0.15475181,   0.09717491,
         -0.19093694,  -0.05085686,  -0.82300156,  -0.32161304],
       [  1.1796119 ,   0.05886428,  -0.00609161,  10.11358   ,
         -0.31034935,  -0.07316032,  -0.10285831,   0.10503297],
       [  0.01057342,   0.01063731,   0.00498417,  -0.05750106,
          0.0243123 ,   0.10422336,  -0.18605901,  -0.1672897 ],
       [  0.46388248,   0.01027704, -14.44516   ,   0.00504395,
         -0.05905372,   1.5797353 ,   0.02550914,  -0.6289146 ]],
      dtype=float32)>
>>> sgd.get_gradients(loss,model.weights)
Traceback (most recent call last):
  File "<pyshell#1071>", line 1, in <module>
    sgd.get_gradients(loss,model.weights)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\optimizers.py", line 93, in get_gradients
    raise ValueError('An operation has `None` for gradient. '
ValueError: An operation has `None` for gradient. Please make sure that all of your ops have a gradient defined (i.e. are differentiable). Common ops without gradient: K.argmax, K.round, K.eval.
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 5s - loss: nan - val_loss: nan
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> loss=model.get_losses_for(x_train)
>>> loss.shape
Traceback (most recent call last):
  File "<pyshell#1074>", line 1, in <module>
    loss.shape
AttributeError: 'list' object has no attribute 'shape'
>>> len(loss)
0
>>> model.fit(x_train,y_train,epochs=1,verbose=2,validation_data=(x_test, y_test))
Train on 8000 samples, validate on 2000 samples
Epoch 1/1
 - 3s - loss: nan - val_loss: nan
<keras.callbacks.callbacks.History object at 0x000001F66BC727B8>
>>> del g
Traceback (most recent call last):
  File "<pyshell#1077>", line 1, in <module>
    del g
NameError: name 'g' is not defined
>>> data = [[(random()*2-1)*m,(random()*2-1)*m] for _ in range(10000)]
>>> x_train = np.array(data[:Nt])
>>> inputs = tf.convert_to_tensor(x_train,float32)
Traceback (most recent call last):
  File "<pyshell#1080>", line 1, in <module>
    inputs = tf.convert_to_tensor(x_train,float32)
NameError: name 'float32' is not defined
>>> inputs = tf.convert_to_tensor(x_train,tf.float32)
>>> inputs[:10]
<tf.Tensor: id=5505, shape=(10, 2), dtype=float32, numpy=
array([[ 0.15520398, -0.69467944],
       [-0.5595289 , -0.7384567 ],
       [ 0.29876515, -0.72500724],
       [-0.7027884 ,  0.765573  ],
       [ 0.8528633 , -0.9616452 ],
       [ 0.2019113 , -0.42200688],
       [ 0.50221956, -0.680707  ],
       [ 0.5110155 ,  0.8681999 ],
       [-0.5782707 ,  0.1706299 ],
       [-0.58954513, -0.6416744 ]], dtype=float32)>
>>> inputs.shape
TensorShape([8000, 2])
>>> with tf.GradientTape(persistent=True) as g:
	g.watch(inputs)
	output = tf.stack(inputs,tf.zeros_like(inputs),axis=2)

	
Traceback (most recent call last):
  File "<pyshell#1087>", line 3, in <module>
    output = tf.stack(inputs,tf.zeros_like(inputs),axis=2)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\dispatch.py", line 180, in wrapper
    return target(*args, **kwargs)
TypeError: stack() got multiple values for argument 'axis'
>>> del g
>>> with tf.GradientTape(persistent=True) as g:
	g.watch(inputs)
	output = tf.stack((inputs,tf.zeros_like(inputs)),axis=2)

	
>>> dy_dx = g.gradient(output, inputs)
>>> dy_dx[:10]
<tf.Tensor: id=5518, shape=(10, 2), dtype=float32, numpy=
array([[1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)>
>>> output.shape
TensorShape([8000, 2, 2])
>>> dy_dx.shape
TensorShape([8000, 2])
>>> J = g.jacobian(output, inputs)
Traceback (most recent call last):
  File "<pyshell#1095>", line 1, in <module>
    J = g.jacobian(output, inputs)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\backprop.py", line 1081, in jacobian
    parallel_iterations=parallel_iterations)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\parallel_for\control_flow_ops.py", line 164, in pfor
    return f()
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 1335, in __call__
    return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 589, in _filtered_call
    (t for t in nest.flatten((args, kwargs), expand_composites=True)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\eager\execute.py", line 61, in quick_execute
    num_outputs)
KeyboardInterrupt
>>> del g
>>> with tf.GradientTape(persistent=True) as g:
	g.watch(inputs)
	output = tf.stack((inputs,inputs),axis=2)

	
>>> dy_dx = g.gradient(output, inputs)
>>> dy_dx.shape
TensorShape([8000, 2])
>>> del g
>>> with tf.GradientTape(persistent=True) as g:
	g.watch(inputs)
	output = tf.add(inputs[:,0],inputs[:1])

	
Traceback (most recent call last):
  File "<pyshell#1103>", line 3, in <module>
    output = tf.add(inputs[:,0],inputs[:1])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\gen_math_ops.py", line 392, in add
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [8000] vs. [1,2] [Op:Add]
>>> del g
>>> with tf.GradientTape(persistent=True) as g:
	g.watch(inputs)
	output = tf.add(inputs[:,0],inputs[:,1])

	
>>> dy_dx = g.gradient(output, inputs)
>>> dy_dx.shape
TensorShape([8000, 2])
>>> dy_dx[:10]
<tf.Tensor: id=5651, shape=(10, 2), dtype=float32, numpy=
array([[1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)>
>>> output[:1-]
SyntaxError: invalid syntax
>>> output[:10]
<tf.Tensor: id=5656, shape=(10,), dtype=float32, numpy=
array([-0.53947544, -1.2979856 , -0.42624208,  0.06278461, -0.10878187,
       -0.22009557, -0.17848742,  1.3792154 , -0.40764076, -1.2312195 ],
      dtype=float32)>
>>> output.shape
TensorShape([8000])
>>> model.weights
[<tf.Variable 'power_dense_1/kernel:0' shape=(4, 8) dtype=float32, numpy=
array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(8, 8) dtype=float32, numpy=
array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32)>, <tf.Variable 'power_dense_2/kernel:0' shape=(16, 8) dtype=float32, numpy=
array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(8, 1) dtype=float32, numpy=
array([[nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan]], dtype=float32)>]
>>> model.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
>>> model.weights
[<tf.Variable 'power_dense_1/kernel:0' shape=(4, 8) dtype=float32, numpy=
array([[ 3.2069550e-03, -1.1357396e-04, -6.9527016e-03, -5.1805689e-03,
        -1.4445622e-02, -3.8479296e-03, -3.9537279e-03,  3.7556789e-03],
       [-1.9642238e-02,  3.4684758e-03,  1.1763723e-02, -1.3098445e-02,
         4.1830693e-03,  1.0895804e-02, -1.1018531e-02,  3.5652256e-04],
       [ 9.9160403e-01,  1.0050846e+00,  1.9982930e+00,  1.0189099e-02,
        -8.0380468e-03, -6.8342225e-03,  7.8421198e-03, -5.2401866e-03],
       [-5.7784184e-03,  9.9271554e-01,  4.5048525e-03,  1.9996377e+00,
        -2.7237942e-02, -6.0789743e-03, -1.2852335e-02,  9.8998053e-04]],
      dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(8, 8) dtype=float32, numpy=
array([[ 9.9736023e-01, -4.1965377e-03,  4.9315607e-03, -2.2634442e-03,
        -1.8519543e-02,  3.2882297e-03, -3.6496797e-03, -7.3533007e-03],
       [ 8.2368134e-03,  9.9069566e-01, -7.1257658e-05,  7.9768598e-03,
        -1.9190729e-02, -1.3036960e-02,  1.0337236e-02, -1.7170267e-02],
       [-1.1133366e-03, -1.9681450e-02,  9.9912763e-01, -7.8873169e-03,
        -1.2393865e-02,  8.0719739e-03,  6.7485636e-03, -9.0330997e-03],
       [-4.1388157e-03,  1.0110962e-02,  1.0050217e+00, -1.0966712e-02,
        -1.1436669e-02, -3.4128949e-03, -1.3453694e-02, -4.8216078e-03],
       [ 1.6010471e-03, -1.9260483e-03, -1.2667235e-02, -1.8040024e-02,
        -1.2354151e-03, -7.3336777e-03, -1.1142666e-02,  4.3684053e-03],
       [-1.3962508e-02, -5.9659816e-03, -3.6995662e-03, -5.7791113e-03,
         1.9359298e-03, -1.5605792e-02, -4.1058417e-03, -1.4526023e-03],
       [ 2.2856959e-03, -2.2971490e-04, -1.3608391e-03,  1.4589871e-03,
        -6.1927061e-03,  1.3199065e-02, -9.1785602e-03, -6.7791492e-03],
       [ 1.1682508e-02, -9.0078758e-03, -6.5715932e-03, -2.3999382e-02,
        -5.3321170e-03, -6.1284942e-03,  6.7811157e-04,  2.4836592e-04]],
      dtype=float32)>, <tf.Variable 'power_dense_2/kernel:0' shape=(16, 8) dtype=float32, numpy=
array([[ 5.12821833e-03, -9.90228169e-03, -7.18792435e-03,
         4.83428640e-03,  1.46866753e-03, -6.33987924e-03,
        -1.22055542e-02, -8.29944573e-03],
       [ 8.84273201e-02, -7.66020361e-03, -1.54715951e-03,
        -1.58178546e-02,  8.62813462e-03, -3.94914392e-03,
         5.40742697e-03, -3.87185160e-03],
       [ 1.51042491e-02,  1.77974999e-03, -3.90710868e-03,
        -1.53343123e-03, -1.91116780e-02, -3.30330711e-03,
        -1.03941113e-02, -1.37480637e-02],
       [ 8.38053599e-03,  2.12721108e-03, -8.99857376e-03,
        -4.20540990e-03, -8.50789994e-03, -2.05992740e-02,
         6.38362672e-03, -1.15864668e-02],
       [ 5.20791858e-03, -9.31830518e-03, -1.05958777e-02,
        -7.36364722e-03,  5.78716537e-03, -1.28259305e-02,
         2.17863964e-03, -1.40511934e-02],
       [ 8.36362969e-03,  7.42862606e-03, -1.52975451e-02,
        -2.66753957e-02, -9.55889933e-03, -1.24818133e-02,
        -4.21570940e-03,  8.91667604e-03],
       [ 7.15469383e-03, -5.47297578e-03,  1.16466684e-02,
         4.93848987e-04, -7.03067286e-03, -1.07362522e-02,
         1.23792468e-02, -9.94726736e-03],
       [-7.53593259e-03, -5.89777343e-03, -1.81399602e-02,
        -1.92514844e-02,  1.00582000e-02, -2.64412090e-02,
        -6.68219943e-03,  9.12499614e-03],
       [ 9.98602450e-01,  1.76751273e-04, -1.58884451e-02,
        -6.09409669e-03, -3.98310274e-03,  5.56474319e-03,
        -6.31123362e-03, -5.81871625e-03],
       [ 6.46660989e-03, -2.61895545e-03, -1.21827424e-02,
        -1.06760301e-02, -3.29487491e-03, -7.55305728e-03,
        -1.18042035e-02, -3.99581669e-03],
       [-2.11711740e-03,  4.85845029e-01, -1.00210579e-02,
         5.48269367e-03,  2.19634525e-03,  1.10700307e-03,
        -1.37571171e-02, -2.59821657e-02],
       [-9.48077161e-03, -3.72961164e-03, -4.42196988e-03,
         7.10813375e-03,  2.78083200e-04, -6.68239547e-03,
        -2.09502168e-02, -2.41766404e-02],
       [-1.71350744e-02,  1.25916125e-02,  1.61754037e-03,
        -4.76043299e-03,  5.12256380e-03, -6.76764827e-03,
        -1.31380092e-02, -6.87167794e-03],
       [ 1.92131177e-02, -3.85988806e-03,  6.10313495e-04,
        -1.45523145e-03, -2.04893621e-03, -8.07377789e-03,
        -1.43021001e-02, -9.82081518e-03],
       [ 1.85189000e-03, -8.76024459e-03, -5.12845069e-03,
        -1.39168119e-02,  1.38318297e-02, -1.39311915e-02,
        -8.28657020e-03, -9.99051705e-03],
       [-8.40454828e-03,  3.72002542e-05,  1.64976567e-02,
        -2.40731761e-02, -1.36988396e-02, -1.52009551e-03,
        -1.20908795e-02,  2.71852035e-03]], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(8, 1) dtype=float32, numpy=
array([[ 0.987181  ],
       [ 0.9858297 ],
       [-0.01525498],
       [ 0.00404392],
       [-0.01346154],
       [ 0.00761406],
       [ 0.00297222],
       [-0.00424237]], dtype=float32)>]
>>> sgd.get_weights
<bound method Optimizer.get_weights of <keras.optimizers.SGD object at 0x000001F66A061F60>>
>>> sgd.get_weights()
[500, array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32), array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32), array([[nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan],
       [nan, nan, nan, nan, nan, nan, nan, nan]], dtype=float32), array([[nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan]], dtype=float32)]
>>> sgd.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
Traceback (most recent call last):
  File "<pyshell#1118>", line 1, in <module>
    sgd.set_weights([w+(np.random.randn(*w.shape)-0.5)*0.01 for w in create_weights()])
  File "C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\optimizers.py", line 126, in set_weights
    'of the optimizer (' + str(len(params)) + ')')
ValueError: Length of the specified weight list (4) does not match the number of weights of the optimizer (5)
>>> 
 RESTART: C:\Users\joshm\AppData\Local\Programs\Python\Python37-32\powerdense.py 
Using TensorFlow backend.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2)                 0         
_________________________________________________________________
power_dense_1 (PowerDense)   (None, 8, 2)              32        
_________________________________________________________________
dense_1 (PowerDense)         (None, 8, 2)              64        
_________________________________________________________________
power_dense_2 (PowerDense)   (None, 8, 2)              128       
_________________________________________________________________
dense_2 (PowerDense)         (None, 1, 2)              8         
=================================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From C:\Users\joshm\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 8000 samples, validate on 2000 samples
>>> 
