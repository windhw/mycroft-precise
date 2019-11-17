import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
import numpy as np
from precise.params import pr
from math import floor


def tflite_mfccs(samples, tflitemodel_path ):
    samples = samples.astype(np.float32)
    samples = np.expand_dims(samples, 1)

    if samples.shape[0] > pr.buffer_samples:
        samples = samples[-pr.buffer_samples:,:]

    real_features  =  1 + int(floor((samples.shape[0] - pr.window_samples) / pr.hop_samples))
    need_cut = pr.n_features - real_features;
    print(real_features,need_cut,pr.n_features)

    if samples.shape[0] < pr.buffer_samples:
        samples = np.concatenate([
               samples,
               np.zeros((pr.buffer_samples - samples.shape[0], samples.shape[1]), dtype=np.float32)
           ])
    interpreter = tf.lite.Interpreter(model_path=tflitemodel_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    #print(input_details)
    #print(output_details)
    input_data = samples 
    interpreter.set_tensor(input_details[0]['index'],input_data )

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data[0]
    if need_cut > 0:
        output_data = output_data[:-need_cut,:]
    return output_data
