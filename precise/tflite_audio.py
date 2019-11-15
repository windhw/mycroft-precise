import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
import numpy as np


def tflite_mfccs(samples, tflitemodel_path ):
    samples = np.expand_dims(samples, 1)
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
    return output_data
