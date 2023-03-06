import math
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from tensorflow.python.keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
)
from onnx2tf.utils.colors import Color
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    calc_tf_pooling_pads,
    calc_extra_padding_with_ceil
)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """AveragePool

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

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    auto_pad = graph_node.attrs.get('auto_pad', 'NOTSET')
    ceil_mode = bool(graph_node.attrs.get('ceil_mode', 0))
    count_include_pad = bool(graph_node.attrs.get('count_include_pad', 0))
    kernel_shape = graph_node.attrs['kernel_shape']
    spatial_size = len(kernel_shape)
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    pads = graph_node.attrs.get('pads', [0] * spatial_size * 2)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)

    input_tensor_shape = input_tensor.shape.as_list()
    is_known_shape = None not in input_tensor_shape[1:]
    average_multiplier = [1] * spatial_size * 2
    average_multiplier_begin = [1] * spatial_size
    average_multiplier_end = [1] * spatial_size

    # default tensorflow action is 'SAME_UPPER' mode (extra padding in the end for odd numbers)
    # explicit pad layer is added for tensorflow incompatible cases
    tf_pad_mode = 'VALID'
    is_explicit_padding = False
    tf_pads = calc_tf_pooling_pads(input_shape=input_tensor_shape,
                                   kernel=kernel_shape,
                                   strides=strides)

    func = math.ceil if ceil_mode else math.floor
    output_spatial_shape = [
        func((i + pb + pe - d * (k - 1) - 1) / s + 1)
        for i, pb, pe, k, d, s in zip(input_tensor_shape[1:-1], pads[:len(pads) // 2], pads[len(pads) // 2:],
                                      kernel_shape, dilations, strides)
    ]

    last_stride_starts = [(o - 1) * s for o, s in zip(output_spatial_shape, strides)]

    last_stride_non_zeros = []
    for input_spatial_shape, pads_begin, last_stride_start, kernel, stride, dilation\
            in zip(input_tensor_shape[1:-1], pads[:len(pads) // 2], last_stride_starts, kernel_shape, strides, dilations):

        # check if last stride starts in valid position
        if pads_begin + input_spatial_shape - last_stride_start > 0:
            non_zeros_values = (pads_begin + input_spatial_shape - last_stride_start) // dilation

            effective_kernel = (kernel - 1) * dilation + 1
            if non_zeros_values >= effective_kernel:
                # last stride larger than kernel means there exists dropped stride after
                # current last stride is full of valid values
                non_zeros_values = kernel

        else:
            # sometimes last stride starts from end side padding area when ceil_mode is used
            # in this case, only zeros are included in padding, so it will be dropped
            # instead, last stride except end side padding should be calculated
            non_zeros_values = pads_begin + input_spatial_shape - last_stride_start + stride

        # if last strides start from begin side padding, zeros should be removed
        if last_stride_start < pads_begin:
            non_zeros_values -= pads_begin

        last_stride_non_zeros.append(non_zeros_values)

    # onnx padding value is ignored if auto_pad is not 'NOTSET'
    if auto_pad == 'NOTSET':

        # check if onnx padding is same with tensorflow padding mode 'SAME'
        # this is to avoid flex operations since tflite builtin pooling operator does not support manual padding
        if is_known_shape and pads != [0] * spatial_size * 2 and tf_pads == pads:
            auto_pad = 'SAME_UPPER'
            tf_pad_mode = 'SAME'

        else:
            auto_pad = 'VALID'
            is_explicit_padding = True

            # extra padding may be needed for ceiling
            # this padding is added to end side (right, bottom) only
            if ceil_mode:
                extra_pads = \
                    calc_extra_padding_with_ceil(input_shape=input_tensor_shape[1:-1],
                                                 kernel=kernel_shape,
                                                 pads=pads,
                                                 dilations=dilations,
                                                 strides=strides)
                pads = pads[:len(pads) // 2] + [p + e for p, e in zip(pads[len(pads) // 2:], extra_pads)]

            tf_pads = pads

    elif auto_pad == 'SAME_UPPER':
        tf_pad_mode = 'SAME'

    elif auto_pad == 'SAME_LOWER':
        is_explicit_padding = True

    elif auto_pad == 'VALID':
        tf_pads = [0] * spatial_size * 2

    else:
        error_msg = f'{Color.RED}ERROR:{Color.RESET} ' + \
                    f'Wrong auto_pad parameter in AveragePool: {auto_pad}.'
        raise ValueError(error_msg)

    # add extra pad layer if needed
    if is_explicit_padding and tf_pads != [0] * spatial_size * 2:
        warning_msg = f'{Color.YELLOW}WARNING:{Color.RESET} ' \
                      f'Tensorflow incompatible padding detected. ' \
                      f'Extra pad layer is inserted automatically. '
        print(warning_msg)

        if auto_pad == 'SAME_LOWER':
            # switch the order of pads
            tf_pads = [i for tup in zip(tf_pads[len(tf_pads) // 2:], tf_pads[:len(tf_pads) // 2]) for i in tup]

        if not count_include_pad:
            average_multiplier_begin = [
                k / (k - p) if v + p >= k else k / v
                for v, p, k
                in zip(input_tensor_shape[1:-1], tf_pads[:len(tf_pads) // 2], kernel_shape)
            ]

            average_multiplier_end = [
                k / n if n > 0 else 1
                for p, k, n
                in zip(tf_pads[len(tf_pads) // 2:], kernel_shape, last_stride_non_zeros)
            ]

        average_multiplier = [i for tup in zip(average_multiplier_begin, average_multiplier_end) for i in tup]

        # convert to tensorflow padding format
        tf_pads = [[0, 0]] + \
                  [list(i) for i in zip(tf_pads[:len(tf_pads) // 2], tf_pads[len(tf_pads) // 2:])] + \
                  [[0, 0]]

        padded_tensor = tf.pad(
            tensor=input_tensor,
            paddings=tf_pads,
            mode='CONSTANT',
        )

    else:
        padded_tensor = input_tensor

        if count_include_pad:
            average_multiplier_begin = [
                (k - p) / k if v + p >= k else v / k
                for v, p, k
                in zip(input_tensor_shape[1:-1], tf_pads[:len(tf_pads) // 2], kernel_shape)
            ]
            average_multiplier_end = [
                n / k if n > 0 else 1
                for p, k, n
                in zip(tf_pads[len(tf_pads) // 2:], kernel_shape, last_stride_non_zeros)
            ]
            average_multiplier = [i for tup in zip(average_multiplier_begin, average_multiplier_end) for i in tup]

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    tf_op_type = None
    if len(kernel_shape) == 1:
        pooled_tensor = AveragePooling1D(
            pool_size=kernel_shape,
            strides=strides,
            padding=tf_pad_mode.upper(),
        )(padded_tensor)
        tf_op_type = AveragePooling1D

    elif len(kernel_shape) == 2:
        pooled_tensor = AveragePooling2D(
            pool_size=kernel_shape,
            strides=strides,
            padding=tf_pad_mode.upper(),
        )(padded_tensor)
        tf_op_type = AveragePooling2D

    elif len(kernel_shape) == 3:
        pooled_tensor = AveragePooling3D(
            pool_size=kernel_shape,
            strides=strides,
            padding=tf_pad_mode.upper(),
        )(padded_tensor)
        tf_op_type = AveragePooling3D

    else:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'AveragePool supports only 1D, 2D, and 3D. ' +\
            f'opname: {graph_node.name} Type: AveragePool{len(kernel_shape)}D'
        print(error_msg)
        raise AssertionError(error_msg)

    # tensorflow average pooling needs extra process to get same output with onnx
    # https://github.com/PINTO0309/onnx2tf/issues/124
    if average_multiplier != [1] * spatial_size * 2:
        warning_msg = \
            f'{Color.YELLOW}WARNING:{Color.RESET} ' \
            f'Tensorflow incompatible action detected. ' \
            f'Some additional layers are inserted to reproduce same output. ' \
            f'Please refer to the following link for more information: ' \
            f'https://github.com/PINTO0309/onnx2tf/issues/124'
        print(warning_msg)

        # TODO: body may contain zero padded value, average multiplier need to be calculated for every row, column
        if pooled_tensor.shape[1] >= 2:
            padded_slice_begin = pooled_tensor[:, 0:1, ...] * average_multiplier[0]
            padded_slice_body = pooled_tensor[:, 1:-1, ...]
            padded_slice_end = pooled_tensor[:, -1:, ...] * average_multiplier[1]
            pooled_tensor = tf.concat([padded_slice_begin, padded_slice_body, padded_slice_end], axis=1)
        elif pooled_tensor.shape[1] == 1:
            pooled_tensor = pooled_tensor * average_multiplier[0]

        if len(kernel_shape) >= 2:
            if pooled_tensor.shape[2] >= 2:
                padded_slice_begin = pooled_tensor[:, :, 0:1, ...] * average_multiplier[2]
                padded_slice_body = pooled_tensor[:, :, 1:-1, ...]
                padded_slice_end = pooled_tensor[:, :, -1:, ...] * average_multiplier[3]
                pooled_tensor = tf.concat([padded_slice_begin, padded_slice_body, padded_slice_end], axis=2)
            elif pooled_tensor.shape[2] == 1:
                pooled_tensor = pooled_tensor * average_multiplier[2]

        if len(kernel_shape) >= 3:
            if pooled_tensor.shape[3] >= 2:
                padded_slice_begin = pooled_tensor[:, :, :, 0:1, ...] * average_multiplier[4]
                padded_slice_body = pooled_tensor[:, :, :, 1:-1, ...]
                padded_slice_end = pooled_tensor[:, :, :, -1:, ...] * average_multiplier[5]
                pooled_tensor = tf.concat([padded_slice_begin, padded_slice_body, padded_slice_end], axis=3)
            elif pooled_tensor.shape[3] == 1:
                pooled_tensor = pooled_tensor * average_multiplier[4]

    tf_layers_dict[graph_node_output.name]['tf_node'] = pooled_tensor

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'x': input_tensor,
                    'pool_size': kernel_shape,
                    'strides': strides,
                    'padding': tf_pads if tf_pad_mode != 'same' else tf_pad_mode,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
