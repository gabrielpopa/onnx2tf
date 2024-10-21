import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
from onnx2tf.utils.logging import *

@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Erf

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input.name]['nhwc'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    }

    replace_erf_to_pseudo_erf = "erf" in kwargs['replace_to_pseudo_operators']
    gelu_replace_op_names: dict = kwargs['gelu_replace_op_names']

    # Generation of TF OP
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    before_trans_shape = input_tensor.shape
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    after_trans_shape = input_tensor.shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')


    # Replace with GeLU if available.
    gelu_op_names = [op_name for op_names in gelu_replace_op_names.values() for op_name in op_names]
    enable_gelu = graph_node.name in gelu_op_names

    if not enable_gelu:
        if not replace_erf_to_pseudo_erf:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.math.erf(
                    x=input_tensor,
                    name=graph_node.name,
                )
        else:
            warn("(gp) Pseudo Erf")
            # https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
            x_abs = tf.math.abs(input_tensor)
            sign = tf.sign(input_tensor)

            # (gp) Working
            # a1 =  0.254829592
            # a2 = -0.284496736
            # a3 =  1.421413741
            # a4 = -1.453152027
            # a5 =  1.061405429
            # p  =  0.3275911
            # t = tf.math.divide(1.0,1.0 + p*x_abs)
            # y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*tf.math.exp(-x_abs*x_abs)
            # erf_tensor = sign*y

            # (gp) Suggested to work
            a1 = tf.constant(0.254829592, dtype=input_tensor.dtype)
            a2 = tf.constant(-0.284496736, dtype=input_tensor.dtype)
            a3 = tf.constant(1.421413741, dtype=input_tensor.dtype)
            a4 = tf.constant(-1.453152027, dtype=input_tensor.dtype)
            a5 = tf.constant(1.061405429, dtype=input_tensor.dtype)
            p  = tf.constant(0.3275911, dtype=input_tensor.dtype)
            t = tf.math.divide(1.0, tf.math.add(1.0, tf.math.multiply(p, x_abs)))
            y = tf.math.subtract(
                1.0, 
                tf.math.multiply(
                    (((((tf.math.multiply(a5, t) + a4) * t) + a3) * t) + a2) * t + a1,
                    t
                ) * tf.math.exp(-tf.math.multiply(x_abs, x_abs))
            )
            erf_tensor = tf.math.multiply(sign, y)

            # (gp) Coefficients for the Abramowitz and Stegun approximation
            # t = tf.math.divide(1.0, (1.0 + 0.5 * x_abs))
            # y = 1.0 - t * tf.math.exp(-x_abs*x_abs - 1.26551223 + 
            #         t * (1.00002368 + 
            #         t * (0.37409196 + 
            #         t * (0.09678418 + 
            #         t * (-0.18628806 + 
            #         t * (0.27886807 + 
            #         t * (-1.13520398 + 
            #         t * (1.48851587 + 
            #         t * (-0.82215223 + 
            #         t * 0.17087277)))))))))
            # erf_tensor = sign * y

            tf_layers_dict[graph_node_output.name]['tf_node'] = erf_tensor
        tf_type = tf.math.erf
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(
                input=input_tensor,
                name=graph_node.name,
            )
        tf_type = tf.identity

    # Post-process transpose
    before_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    after_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_type,
                'tf_inputs': {
                    'x': input_tensor,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
