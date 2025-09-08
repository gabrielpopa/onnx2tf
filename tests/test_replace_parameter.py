import pytest
from onnx2tf.utils.common_functions import replace_parameter


def test_replace_parameter_str_true_to_bool():
    result = replace_parameter(
        value_before_replacement=False,
        param_target='attributes',
        param_name='flag',
        op_rep_params=[{'param_target': 'attributes', 'param_name': 'flag', 'values': 'True'}]
    )
    assert result is True


def test_replace_parameter_str_false_to_bool():
    result = replace_parameter(
        value_before_replacement=True,
        param_target='attributes',
        param_name='flag',
        op_rep_params=[{'param_target': 'attributes', 'param_name': 'flag', 'values': 'False'}]
    )
    assert result is False
