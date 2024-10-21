import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from typing import List
from onnx2tf.utils.common_functions import (
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    convert_axis,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)

def dynamic_split_compatible(input_tensor, split_sizes, axis=0):
    split_sizes = tf.convert_to_tensor(split_sizes, dtype=tf.int32)
    # num_splits = split_sizes.shape[0] or tf.shape(split_sizes)[0]
    # indices = tf.cumsum(split_sizes)[:-1]
    return tf.split(input_tensor, num_or_size_splits=split_sizes, axis=axis)

def split_tensor_gpu_compatible(input_tensor, axis, num_or_size_splits):
    # Get the dynamic shape and rank of the input tensor
    input_shape = tf.shape(input_tensor)
    input_tensor_rank = tf.rank(input_tensor)
    
    # Adjust negative axis values
    axis = tf.where(axis >= 0, axis, axis + input_tensor_rank)
    axis_int = axis.numpy() if isinstance(axis, tf.Tensor) and axis.shape == () else axis
    
    # Build the permutation to bring the split axis to the front
    perm_pre = tf.expand_dims(axis_int, 0)
    perm_post = tf.concat([tf.range(axis_int), tf.range(axis_int + 1, input_tensor_rank)], axis=0)
    perm = tf.concat([perm_pre, perm_post], axis=0)
    
    # Transpose the tensor to bring the split axis to the front
    x = tf.transpose(input_tensor, perm)
    
    # Merge the remaining dimensions
    remaining_dims = tf.gather(input_shape, perm_post)
    merged_dim = tf.reduce_prod(remaining_dims)
    x = tf.reshape(x, tf.concat([[input_shape[axis_int]], [merged_dim]], axis=0))
    
    # Expand dimensions to make it 4D
    x_shape = tf.shape(x)
    x = tf.reshape(x, tf.concat([x_shape, [1, 1]], axis=0))  # Now x has shape [split_axis_size, merged_dim, 1, 1]
    
    # Perform the split along the first axis
    split_tensors = tf.split(x, num_or_size_splits=num_or_size_splits, axis=0)
    
    final_tensors = []
    for split_tensor in split_tensors:
        # Remove the extra dimensions added earlier
        split_tensor = tf.squeeze(split_tensor, axis=[2, 3])  # Now shape is [1, merged_dim]
        # Reshape back to the original dimensions (excluding the split axis)
        split_tensor = tf.reshape(split_tensor, remaining_dims)
        # No need to permute back since the dimensions are in the correct order
        final_tensors.append(split_tensor)
    
    return final_tensors

def split_tensor_gpu_compatible3(input_tensor, axis, num_or_size_splits):
    # Get the dynamic shape and rank of the input tensor
    input_shape = tf.shape(input_tensor)
    input_tensor_rank = tf.rank(input_tensor)
    
    # Adjust negative axis values
    axis = tf.where(axis >= 0, axis, axis + input_tensor_rank)
    
    # Build the permutation to bring the split axis to the front
    perm_pre = tf.expand_dims(axis, 0)
    perm_post = tf.concat([tf.range(axis), tf.range(axis + 1, input_tensor_rank)], axis=0)
    perm = tf.concat([perm_pre, perm_post], axis=0)
    
    # Transpose the tensor to bring the split axis to the front
    x = tf.transpose(input_tensor, perm)
    
    # Merge the remaining dimensions
    remaining_dims = tf.gather(input_shape, perm_post)
    merged_dim = tf.reduce_prod(remaining_dims)
    x = tf.reshape(x, tf.concat([[input_shape[axis]], [merged_dim]], axis=0))
    
    # Expand dimensions to make it 4D
    x = tf.reshape(x, tf.concat(tf.shape(x), [1, 1]), axis=0)  # Shape becomes [split_axis_size, merged_dim, 1, 1]
    
    # Perform the split along the first axis
    split_tensors = tf.split(x, num_or_size_splits=num_or_size_splits, axis=0)
    
    final_tensors = []
    for split_tensor in split_tensors:
        # Remove the extra dimensions added earlier
        split_tensor = tf.reshape(split_tensor, [merged_dim])
        # Reshape back to the original dimensions (excluding the split axis)
        split_tensor = tf.reshape(split_tensor, remaining_dims)
        # No need to permute back since the dimensions are in the correct order
        final_tensors.append(split_tensor)
    
    return final_tensors

def split_tensor_gpu_compatible2(input_tensor, axis, num_or_size_splits):
    # Get the dynamic shape of the input tensor
    input_shape = tf.shape(input_tensor)
    input_tensor_rank = input_tensor.shape.rank

    # Move the axis to split to the first dimension
    print(f"axis: {axis}")
    perm = [axis] + [i for i in range(input_tensor_rank) if i != axis]
    print(f"perm: {perm}")
    x = tf.transpose(input_tensor, perm)

    # Combine the remaining dimensions into a single dimension to reduce rank to 4
    remaining_dims = [input_shape[i] for i in perm[1:]]
    merged_dim = tf.reduce_prod(remaining_dims)
    x = tf.reshape(x, tf.concat([[input_shape[axis]], [merged_dim]], axis=0))

    # Expand dimensions to make it a 4D tensor (if necessary)
    x = tf.expand_dims(x, axis=-1)  # Now x has shape [split_axis_size, merged_dim, 1]

    # Perform the split operation along the first axis
    split_tensors = tf.split(x, num_or_size_splits=num_or_size_splits, axis=0)

    final_tensors = []
    for split_tensor in split_tensors:
        # Remove the extra dimension added earlier
        split_tensor = tf.squeeze(split_tensor, axis=-1)
        # Reshape back to the original dimensions (except the split axis)
        split_tensor = tf.reshape(split_tensor, remaining_dims)
        # Permute the dimensions back to their original order
        inverse_perm = [perm.index(i) for i in range(input_tensor_rank)]
        split_tensor = tf.transpose(split_tensor, inverse_perm[1:])
        final_tensors.append(split_tensor)

    return final_tensors

# (gp) bun
def split_without_slice_old(input_tensor, split_sizes, axis=0):
    # Compute the cumulative sum of split sizes to get the split points
    split_points = tf.concat([[0], tf.cumsum(split_sizes)], axis=0)
    
    # Initialize an empty list to hold the split tensors
    splits = []
    
    # Loop over each split size to generate the corresponding indices
    for i in range(len(split_sizes)):
        start = split_points[i]
        end = split_points[i + 1]
        indices = tf.range(start, end)
        
        # Use tf.gather to extract the slices along the specified axis
        split_i = tf.gather(input_tensor, indices, axis=axis)
        splits.append(split_i)
    
    return splits

def split_without_slice(input_tensor, split_sizes, axis=0):
    # Ensure split_sizes is of type int32
    split_sizes = tf.cast(split_sizes, tf.int32)
    
    # Compute the cumulative sum of split sizes to get the split points
    split_points = tf.concat([[0], tf.cumsum(split_sizes)], axis=0)
    
    # Ensure split_points is of type int32
    split_points = tf.cast(split_points, tf.int32)
    
    # Initialize an empty list to hold the split tensors
    splits = []
    
    # Loop over each split size to generate the corresponding indices
    for i in range(len(split_sizes)):
        start = split_points[i]
        end = split_points[i + 1]
        
        # Use tf.range to generate indices and ensure they are int32
        indices = tf.range(start, end, dtype=tf.int32)
        
        # Use tf.gather to extract the slices along the specified axis
        split_i = tf.gather(input_tensor, indices, axis=axis)
        splits.append(split_i)
    
    return splits

def split_using_unstack(input_tensor, split_sizes, axis=0):
    # Unstack the tensor along the specified axis
    unstacked = tf.unstack(input_tensor, axis=axis)
    
    # Initialize list for splits
    splits = []
    index = 0
    for size in split_sizes:
        # Extract the slices for the current split
        split_i = unstacked[index:index + size]
        # Stack them back along the same axis
        split_tensor = tf.stack(split_i, axis=axis)
        splits.append(split_tensor)
        index += size
    return splits

def split_without_slice_high_dim(input_tensor, split_sizes, axis=0):
    # Ensure split_sizes are int32
    split_sizes = tf.cast(split_sizes, tf.int32)
    
    # Get the rank of the input tensor
    input_rank = tf.rank(input_tensor)
    
    # Create a permutation that moves the split axis to the first dimension
    perm = tf.concat([[axis], tf.range(axis), tf.range(axis + 1, input_rank)], axis=0)
    transposed = tf.transpose(input_tensor, perm)
    
    # Flatten the rest of the dimensions
    transposed_shape = tf.shape(transposed)
    reshaped = tf.reshape(transposed, [transposed_shape[0], -1])
    
    # Compute split points
    split_points = tf.concat([[0], tf.cumsum(split_sizes)], axis=0)
    split_points = tf.cast(split_points, tf.int32)
    
    splits = []
    for i in range(len(split_sizes)):
        start = split_points[i]
        end = split_points[i + 1]
        indices = tf.range(start, end, dtype=tf.int32)
        split_i = tf.gather(reshaped, indices, axis=0)
        
        # Reshape back to original dimensions
        split_dim = end - start
        split_shape = tf.concat([[split_dim], transposed_shape[1:]], axis=0)
        split_i = tf.reshape(split_i, split_shape)
        
        # Transpose back to original axis order
        inv_perm = tf.math.invert_permutation(perm)
        split_i = tf.transpose(split_i, inv_perm)
        
        splits.append(split_i)
    
    return splits

@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Split

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = before_op_output_shape_trans_1

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor_rank = len(graph_node_input_1.shape) \
        if graph_node_input_1.shape is not None \
            else len(tf_layers_dict[graph_node_input_1.name]['tf_node'].shape)

    graph_node_input_2 = None
    if len(graph_node.inputs) >= 2:
        graph_node_input_2 = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans if isinstance(graph_node_input_2, gs.Variable) else False,
        )
    # graph_node_output: gs.Variable = graph_node.outputs[0]
    graph_node_outputs: List[gs.Variable] = [
        graph_node_output for graph_node_output in graph_node.outputs
    ]

    shape = graph_node_outputs[0].shape
    dtype = graph_node_outputs[0].dtype
    axis = graph_node.attrs.get('axis', 0)
    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=input_tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    num_outputs = graph_node.attrs.get('num_outputs', None)

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_shape = input_tensor.shape

    split = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if split is not None and split.shape is None:
        split = len(graph_node_outputs)
    if split is None:
        split = len(graph_node_outputs)
    split = graph_node.attrs.get('split', split)

    # (gp) Prevent Flex?
    if (dtype == tf.float16):
        dtype = tf.float32
 
    for graph_node_output in graph_node_outputs:
        # Preserving Graph Structure (Dict)
        tf_layers_dict[graph_node_output.name] = {
            'optype': graph_node.op,
            'shape': shape,
            'dtype': dtype,
            'nhwc': tf_layers_dict[graph_node_input_1.name]['nhwc'] \
                if isinstance(graph_node_input_1, gs.Variable) \
                    and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False
        }

    # Param replacement
    axis = replace_parameter(
        value_before_replacement=axis,
        param_target='attributes',
        param_name='axis',
        **kwargs,
    )
    num_outputs = replace_parameter(
        value_before_replacement=num_outputs,
        param_target='attributes',
        param_name='num_outputs',
        **kwargs,
    )
    split = replace_parameter(
        value_before_replacement=split,
        param_target='inputs',
        param_name='split',
        **kwargs,
    )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    splited_tensors = None
    if (
            (isinstance(split, int) and split == 1) \
                or \
            (isinstance(split, np.ndarray) and len(list(split)) == 1 and split[0] == 1)
        ) \
        and isinstance(input_tensor_shape[axis], int) \
        and input_tensor_shape[axis] == 1:
        # Disable unnecessary splits
        splited_tensors = \
            [
                tf.identity(
                    input=input_tensor,
                    name=graph_node.name,
                )
            ]
    elif isinstance(split, np.ndarray) \
        and len(list(split)) > 1 \
        and np.prod(split) == 1 \
        and isinstance(input_tensor_shape[axis], int) \
        and input_tensor_shape[axis] == len(list(split)):

        splited_tensors = split_using_unstack(input_tensor, split, axis)
        

    elif isinstance(split, np.ndarray) \
        and len(list(split)) > 1 \
        and np.prod(split) != 1 \
        and isinstance(input_tensor_shape[axis], int) \
        and len(split) == sum([1 for dim in split if isinstance(dim, np.int64) or isinstance(dim, int)]) \
        and len(split) == sum([1 for dim in split if split[0] == dim]):
        # Suppression of FlexSplitV generation
        splited_tensors = \
            tf.split(
                value=input_tensor,
                num_or_size_splits=len(split),
                axis=axis,
                num=None,
                name=graph_node.name,
            )
    elif isinstance(split, np.ndarray) \
        and len(list(split)) > 1 \
        and np.prod(split) != 1 \
        and isinstance(input_tensor_shape[axis], int) \
        and len(split) == sum([1 for dim in split if isinstance(dim, np.int64) or isinstance(dim, int)]) \
        and len(split) != sum([1 for dim in split if split[0] == dim]) \
        and np.sum(split) == input_tensor_shape[axis]:
        # Suppression of FlexSplitV generation
        # SplitV -> Strided_Slice
        splited_tensors = []
        begin_stock = []
        for split_idx, split_dim in enumerate(split):
            begin_ = []
            end_ = []
            begin_mask_ = 0
            end_mask_ = 0
            for idx in range(input_tensor_rank):
                if idx == axis:
                    if split_idx == 0:
                        begin_.append(0)
                    else:
                        begin_.append(begin_stock[split_idx-1][axis] + split[split_idx-1])
                    end_.append(begin_[-1] + split_dim)
                else:
                    begin_.append(0)
                    end_.append(0)
                    begin_mask_ = begin_mask_ + 2**idx
                    end_mask_ = end_mask_ + 2**idx

            exit(3)
            splited_tensors.append(
                tf.strided_slice(
                    input_=input_tensor,
                    begin=begin_,
                    end=end_,
                    begin_mask=begin_mask_,
                    end_mask=end_mask_,
                )
            )
            begin_stock.append(begin_)
    else:
        splited_tensors = dynamic_split_compatible(input_tensor, split, axis)
        
    for splited_tensor, graph_node_output in zip(splited_tensors, graph_node_outputs):
        tf_layers_dict[graph_node_output.name]['tf_node'] = splited_tensor
        # Post-process transpose
        tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node_output.name,
            **kwargs,
        )

    # Generation of Debug Info
    tf_outputs = {f"output{idx}": value for idx, value in enumerate(splited_tensors)}
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.split,
                'tf_inputs': {
                    'value': input_tensor,
                    'num_or_size_splits': split,
                    'axis': axis,
                    'num': num_outputs,
                },
                'tf_outputs': tf_outputs,
            }
        )
