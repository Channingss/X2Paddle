# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from x2paddle.decoder.onnx_decoder import ONNXGraph, ONNXGraphNode, ONNXGraphDataNode
from x2paddle.core.graph import GraphNode
from x2paddle.core.fluid_code import Layer
from x2paddle.core.fluid_code import FluidCode
from x2paddle.core.util import string
from x2paddle.op_mapper.onnx2paddle.opset9.custom_layer import *
from functools import reduce
import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import logging as _logging
from collections import OrderedDict
import math
import os
import shutil

_logger = _logging.getLogger(__name__)


def _const_weight_or_none(node):
    if 'Constant' in node.layer_type:
        return node.value
    if isinstance(node, ONNXGraphDataNode):
        return node.weight
    return None


def _get_same_padding(in_size, kernel_size, stride):
    new_size = int(math.ceil(in_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + kernel_size - in_size
    pad0 = int(pad_size / 2)
    pad1 = pad_size - pad0
    return [pad0, pad1]


def print_mapping_info(func):
    def run_mapping(*args, **kwargs):
        node = args[1]
        try:
            res = func(*args, **kwargs)
        except:
            print("convert failed node:{}, op_type is {}".format(
                node.layer_name[9:], node.layer_type))
            raise
        else:
            #print("convert successfully node:{}, op_type is {}".format(
            #    node.layer_name[9:], node.layer_type))
            return res

    return run_mapping


class OpSet9():
    elementwise_ops = {
        'Add': 'elementwise_add',
        'Div': 'elementwise_div',
        'Sub': 'elementwise_sub',
        'Mul': 'elementwise_mul',
        'Pow': 'elementwise_pow',
    }

    default_op_mapping_field_values = OrderedDict()
    default_op_mapping_field_values['FLUID_OP'] = ''
    default_op_mapping_field_values['FLUID_INPUT_ARGS'] = None
    default_op_mapping_field_values['FLUID_OUTPUT_ARGS'] = None
    default_op_mapping_field_values['ATTR_MAPPING'] = dict()
    default_op_mapping_field_values['DEFAULTS'] = dict()
    default_op_mapping_field_values['INPUT_PERM'] = None
    default_op_mapping_field_values['OUTPUT_PERM'] = None
    default_op_mapping_field_values['FILL_NAME_FIELD'] = True

    default_op_mapping = {
        'Shape': ['shape', ['X'], ['Out']],
        'Clip': [
            'clip', ['X'], ['Out'], dict(), dict(
                min=(np.asarray(
                    [255, 255, 127, 255], dtype=np.uint8).view(np.float32)[0]),
                max=(np.asarray(
                    [255, 255, 127, 127], dtype=np.uint8).view(np.float32)[0]),
            )
        ],
        'Erf': ['erf', ['X'], ['Out']],
        'Ceil': ['ceil', ['X'], ['Out']],
        'ReduceMean': [
            'reduce_mean', ['X'], ['Out'], dict(
                axes='dim', keepdims='keep_dim'), dict(keep_dim=1)
        ],
        'ReduceSum': [
            'reduce_sum', ['X'], ['Out'], dict(
                axes='dim', keepdims='keep_dim'), dict(keep_dim=1)
        ],
        'ReduceMin': [
            'reduce_min', ['X'], ['Out'], dict(
                axes='dim', keepdims='keep_dim'), dict(keep_dim=1)
        ],
        'ReduceMax': [
            'reduce_max', ['X'], ['Out'], dict(
                axes='dim', keepdims='keep_dim'), dict(keep_dim=1)
        ],
        #active function
        'Relu': ['relu', ['X'], ['Out']],
        'LeakyRelu': ['leaky_relu', ['X'], ['Out'], dict(), dict(alpha=.01)],
        'Elu': ['elu', ['X'], ['Out'], dict(), dict(alpha=1.)],
        'ThresholdedRelu': [
            'thresholded_relu', ['X'], ['Out'], dict(alpha='threshold'),
            dict(alpha=1.)
        ],
        'Tanh': ['tanh', ['X'], ['Out']],
        'Sigmoid': ['sigmoid', ['X'], ['Out']],
        'HardSigmoid': [
            'hard_sigmoid', ['X'], ['Out'], dict(
                alpha='slope', beta='offset'), dict(
                    slope=.2, offset=.5)
        ],
        'Softsign': ['softsign', ['X'], ['Out']],
        'Softplus': ['softplus', ['X'], ['Out']],
        'Exp': ['exp', ['X'], ['Out']],
        'Softmax': ['softmax', ['X'], ['Out'], dict(), dict(axis=1)],
        'Sqrt': ['sqrt', ['X'], ['Out']],
        'Floor': ['floor', ['X'], ['Out']],
        'Abs': ['abs', ['X'], ['Out']],
    }

    default_ioa_constraint = {}

    def __init__(self, decoder):
        super(OpSet9, self).__init__()
        self.graph = decoder.graph
        self.input_shapes = []
        self.weights = dict()
        self.omit_nodes = list()
        self.used_custom_layers = dict()

    @print_mapping_info
    def directly_map(self, node, name='', *args, **kwargs):
        inputs = node.layer.input
        outputs = node.layer.output
        op_type = node.layer_type
        attrs = node.attr_map
        info = self.default_op_mapping[op_type]
        info.extend(
            list(self.default_op_mapping_field_values.values())[len(info):])
        (
            fluid_op,
            fluid_input_args,
            fluid_output_args,
            attr_mapping,
            default_attrs,
            input_perm,
            output_perm,
            fill_name_field, ) = info

        if fluid_op in self.default_ioa_constraint:
            for predicate, message in self.default_ioa_constraint[fluid_op]:
                assert predicate(inputs, outputs, attrs), message

        mapped_attrs = {
            attr_mapping.get(key, key): value
            for key, value in attrs.items()
        }
        if '' in mapped_attrs:
            mapped_attrs.pop('')
        if '_' in mapped_attrs:
            mapped_attrs.pop('_')
        fluid_attrs = default_attrs.copy()
        fluid_attrs.update(mapped_attrs)
        inputs = inputs if input_perm is None else list(
            map(lambda i: inputs[i], input_perm))
        val_inps = []
        for idx, ipt in enumerate(inputs):
            val_inps.append(self.graph.get_input_node(node, idx=idx, copy=True))

        val_outs = outputs if output_perm is None else list(
            map(lambda i: outputs[i], output_perm))
        attr = fluid_attrs
        assert len(val_inps) == 1, 'directly_map error with multi inputs'
        if fluid_op not in ['shape', 'erf']:
            attr['name'] = string(node.layer_name)
        node.fluid_code.add_layer(
            fluid_op, inputs=val_inps[0], output=val_outs[0], param_attr=attr)
        if fluid_op in ['shape']:
            node.fluid_code.add_layer(
                'cast',
                inputs=val_outs[0],
                output=val_outs[0],
                param_attr={'dtype': string('int64')})

    @print_mapping_info
    def deal_custom_layer(self, node):
        op = node.layer_type
        custom_code, func = make_custom_layer(node)
        child_func_code, child_func = make_custom_child_func(node)
        params = get_params(node.layer, node.layer_type)
        arg_names, kwargs = set_args(func, params)
        kwargs['name'] = string(node.layer_name)
        node.fluid_code.add_layer(
            func.__code__.co_name,
            inputs=node.inputs,
            output=node,
            param_attr=kwargs,
            is_custom_layer=True)
        if op not in self.used_custom_layers:
            self.used_custom_layers[op] = custom_code
            if op + '_child_func' not in self.used_custom_layers:
                if child_func_code is not None:
                    self.used_custom_layers[op +
                                            '_child_func'] = child_func_code

    @print_mapping_info
    def elementwise_map(self, node):
        assert node.layer_type in self.elementwise_ops
        op_type = self.elementwise_ops[node.layer_type]

        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        val_y_shape = val_y.out_shapes[0]
        val_x_shape = val_x.out_shapes[0]

        if len(val_x_shape) < len(val_y_shape):
            val_x, val_y = val_y, val_x
            val_y_shape, val_x_shape = val_x_shape, val_y_shape

        str_y_shape = ','.join(str(e) for e in val_y_shape)
        str_x_shape = ','.join(str(e) for e in val_x_shape)
        slice_idx = 0
        if str_y_shape not in str_x_shape:
            for dim in val_y_shape:
                if dim == 1:
                    slice_idx += 1
                else:
                    break
        attr = {"name": string(node.layer_name)}
        if slice_idx < len(val_y_shape) and slice_idx > 0:
            val_y_reshaped = val_y_shape[slice_idx:]
            var_y_reshaped = val_y.layer_name + '_reshaped'
            attr_reshaped = {
                'shape': val_y_reshaped,
                'name': string(var_y_reshaped)
            }
            node.fluid_code.add_layer(
                'reshape',
                inputs=val_y,
                output=var_y_reshaped,
                param_attr=attr_reshaped)
            inputs = {'x': val_x, 'y': var_y_reshaped}
            node.fluid_code.add_layer(
                op_type, inputs=inputs, output=node, param_attr=attr)
        else:
            inputs = {'x': val_x, 'y': val_y}
            node.fluid_code.add_layer(
                op_type, inputs=inputs, output=node, param_attr=attr)

    @print_mapping_info
    def place_holder(self, node):
        self.input_shapes.append(node.out_shapes[0])

        shape = node.out_shapes[0]
        for i, dim_shape in enumerate(shape):
            if dim_shape == 0 and i == 0:
                shape[i] = 1
            if dim_shape == 0 and i != 0:
                assert 'shape of input is not assigned'
        attr = {
            "dtype": string(node.dtype),
            "shape": shape,
            "name": string(node.layer_name),
            "append_batch_size": 'False'
        }

        node.fluid_code.add_layer(
            "data", inputs=None, output=node, param_attr=attr)

    @print_mapping_info
    def create_parameter(self, node, parameter=None):
        if parameter is not None:
            node = parameter
        dtype = node.dtype
        shape = node.out_shapes[0]
        if len(node.weight.shape) == 0:
            shape = [1]
        self.weights[node.layer_name] = node.weight
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name),
            'default_initializer': 'Constant(0.0)'
        }
        if dtype == 'bool':
            attr['dtype'] = string('int64')
            node.fluid_code.add_layer(
                "create_parameter", inputs=None, output=node, param_attr=attr)
            node.fluid_code.add_layer(
                "cast",
                inputs=node,
                output=node,
                param_attr={'dtype': string('bool')})
        elif dtype == 'uint8':
            attr['dtype'] = string('float32')
            node.fluid_code.add_layer(
                "create_parameter", inputs=None, output=node, param_attr=attr)
        else:
            node.fluid_code.add_layer(
                "create_parameter", inputs=None, output=node, param_attr=attr)

    def _pad_if_asymmetric(self, node, pads, val_name):  # pads: SSEE
        assert len(pads) & 1 == 0
        symmetric = True
        ndims = len(pads) // 2
        for idx_dim in range(ndims):
            if pads[idx_dim] != pads[ndims + idx_dim]:
                symmetric = False
                break
        if symmetric:
            return pads[:ndims], val_name
        val_padded = self.Pad(node, op_independent=False)
        return [0] * ndims, val_padded

    def _interpolate(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        if node.layer_type == 'Resize':
            val_scales = self.graph.get_input_node(node, idx=2, copy=True)
        elif node.layer_type == 'Upsample':
            val_scales = self.graph.get_input_node(node, idx=1, copy=True)

        attr = {'name': string(node.layer_name)}
        mode = node.get_attr('mode', 'nearest')
        fluid_op = 'resize_{}'.format(mode)
        if 'linear' in mode:
            print(
                'Warnning: paddle not support op:resize wiht mode: linear, we use bilinear replace linear'
            )
            fluid_op = 'resize_bilinear'

        node.fluid_code.add_layer(
            fluid_op,
            inputs={'input': val_x,
                    'scale': val_scales},
            output=node,
            param_attr=attr)

    @print_mapping_info
    def RoiAlign(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_rois = self.graph.get_input_node(node, idx=1, copy=True)

        pooled_height = node.get_attr('output_height')
        pooled_width = node.get_attr('output_width')
        spatial_scale = node.get_attr('spatial_scale')
        sampling_ratio = node.get_attr('sampling_ratio')
        attr = {
            'pooled_height': pooled_height,
            'pooled_width': pooled_width,
            'spatial_scale': spatial_scale,
            'sampling_ratio': sampling_ratio,
        }
        node.fluid_code.add_layer(
            'roi_align',
            inputs={'input': val_x,
                    'rois': val_rois},
            output=node,
            param_attr=attr)

    @print_mapping_info
    def MaxRoiPool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_rois = self.graph.get_input_node(node, idx=1, copy=True)

        spatial_scale = node.get_attr('spatial_scale')
        pooled_height, pooled_width = node.get_attr('pooled_shape')
        attr = {
            'pooled_height': pooled_height,
            'pooled_width': pooled_width,
            'spatial_scale': spatial_scale,
        }
        node.fluid_code.add_layer(
            'roi_pool',
            inputs={'input': val_x,
                    'rois': val_rois},
            output=node,
            param_attr=attr)

    @print_mapping_info
    def Pad(self, node, op_independent=True):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        pads = node.get_attr('pads')
        mode = node.get_attr('mode', 'constant')
        value = node.get_attr('value', 0.)
        data_shape = val_x.out_shapes[0]
        output_shape = node.out_shapes[0]
        assume_pad2d = False
        attr = {}
        if len(pads) == 4:
            assume_pad2d |= mode != 'constant'
            if data_shape:
                assume_pad2d |= data_shape and len(data_shape) == 4  # NCHW
            if output_shape:
                assume_pad2d |= output_shape and len(output_shape) == 4  # NCHW
        if assume_pad2d:
            fluid_op = 'pad2d'
            attr['data_format'] = string('NCHW')
            attr['mode'] = string(mode)
        else:
            attr = {'pad_value': value}
            fluid_op = 'pad'
        if len(pads) == 4:
            paddings = np.array(pads).reshape(
                (-1, 2)).transpose().flatten().tolist()  # SSEE -> SESE
        elif len(pads) == 8:
            paddings = np.array(pads).reshape(
                (-1, 4)).transpose().flatten().tolist()  # SSEE -> SESE
            if sum(paddings[:4]) == 0:
                fluid_op = 'pad2d'
                paddings = paddings[4:]
                attr['mode'] = string(mode)
        attr['paddings'] = paddings
        if op_independent:
            attr['name'] = string(node.layer_name)
            node.fluid_code.add_layer(
                fluid_op, inputs=val_x, output=node, param_attr=attr)
        else:
            attr['name'] = string(node.layer_name + '_paded')
            node.fluid_code.add_layer(
                fluid_op,
                inputs=val_x,
                output=node.layer_name + '_paded',
                param_attr=attr)
            return node.layer_name + '_paded'

    @print_mapping_info
    def Unsqueeze(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        attr = {'axes': axes, 'name': string(node.layer_name)}
        if len(val_x.out_shapes[0]) == 0:
            if node.layer_name:
                node.fluid_code.add_layer(
                    'reshape',
                    inputs=val_x,
                    output=node,
                    param_attr={'shape': [1]})
        else:
            node.fluid_code.add_layer(
                'unsqueeze', inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def Shrink(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        bias = node.get_attr('bias')
        lambd = node.get_attr('lambd')
        assert bias == 0.0, 'not support bias!=0'
        attr = {'threshold': lambd, 'name': node.layer_name}
        node.fluid_code.add_layer(
            'hard_shrink', inputs=val_x, output=node, param_attr=attr)

    def Greater(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        node.fluid_code.add_layer(
            'greater_than',
            inputs={'x': val_x,
                    'y': val_y},
            output=node,
            param_attr=None)

    @print_mapping_info
    def Constant(self, node):
        val_output = self.graph.get_node(node.layer.output[0], copy=True)

        value = node.get_attr('value')
        dtype = np.dtype(value.dtype)
        output_dtype = val_output.dtype
        if output_dtype:
            assert dtype == output_dtype, 'tensor dtype unmatches storage dtype'

        shape = node.get_attr('shape', None)

        if shape is None:
            shape = val_output.out_shapes[0]
        if shape is None:
            shape = list(value.shape)
            _logger.warning('in (Constant -> %s): '
                            'attribute "shape" of %s not inferred, '
                            'using value as 1-D tensor may lead to fails',
                            val_output.layer_name, val_output.layer_name)

        if len(value) == 1:
            value = value.tolist()
            shape = [1]
            value = value[0]
            if dtype.name == 'int64':
                dtype = 'int32'
            attr = {'shape': shape, 'dtype': string(dtype), 'value': value}
            node.fluid_code.add_layer(
                'fill_constant', inputs=None, output=node, param_attr=attr)
        else:
            if dtype.name == 'uint8':
                dtype = 'int64'
            value = np.reshape(value, shape)
            self.weights[node.layer_name] = value
            attr = {
                'dtype': string(dtype),
                'shape': shape,
                'name': string(node.layer_name),
                'default_initializer': 'Constant(0.0)'
            }
            node.fluid_code.add_layer(
                "create_parameter", inputs=None, output=node, param_attr=attr)

    @print_mapping_info
    def Resize(self, node):
        self._interpolate(node)

    @print_mapping_info
    def Upsample(self, node):
        self._interpolate(node)

    @print_mapping_info
    def InstanceNormalization(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_scale = self.graph.get_input_node(node, idx=1, copy=True)
        val_b = self.graph.get_input_node(node, idx=2, copy=True)
        epsilon = node.get_attr('epsilon', 1e-5)
        attr = {
            'epsilon': epsilon,
            'param_attr': string(val_scale.layer_name),
            'bias_attr': string(val_b.layer_name)
        }
        node.fluid_code.add_layer(
            "instance_norm", inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def Expand(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_shape = self.graph.get_input_node(node, idx=1, copy=True)

        if len(val_shape.outputs) == 1:
            self.omit_nodes.append(val_shape.layer_name)

        val_y = self.graph.get_node(node.layer.output[0], copy=True)
        out_shape = node.out_shapes[0]
        val_x_dtype = val_x.dtype

        name_ones = node.layer_name + '_ones'
        attr_ones = {'shape': out_shape, 'dtype': string(val_x_dtype)}
        node.fluid_code.add_layer(
            'ones', inputs=None, output=name_ones, param_attr=attr_ones)
        inputs = {'x': name_ones, 'y': val_x}
        attr = {'name': string(node.layer_name)}
        node.fluid_code.add_layer(
            'elementwise_mul',
            inputs=inputs,
            output=node.layer_name,
            param_attr=attr)

    @print_mapping_info
    def Gather(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        indices = self.graph.get_input_node(node, idx=1, copy=True)
        indices_shape = indices.out_shapes[0]
        axis = node.get_attr('axis', 0)
        #assert len(
        #    indices_shape) <= 2, "Gather op don't support dim of indice >2 "
        if axis == 0 and len(indices_shape) <= 1:
            node.fluid_code.add_layer(
                'gather',
                inputs={'input': val_x,
                        'index': indices},
                output=node,
                param_attr=None)
        elif axis > 0 and len(indices_shape) <= 1:
            perm = list(range(len(val_x.out_shapes[0])))
            perm = [axis] + perm[:axis] + perm[axis + 1:]
            attr_trans = {'perm': perm}
            name_trans = val_x.layer_name + '_trans'
            node.fluid_code.add_layer(
                'transpose',
                inputs=val_x,
                output=name_trans,
                param_attr=attr_trans)
            node.fluid_code.add_layer(
                'gather',
                inputs={'input': name_trans,
                        'index': indices},
                output=node,
                param_attr=None)
            node.fluid_code.add_layer(
                'transpose', inputs=node, output=node, param_attr=attr_trans)
        elif axis == 0 and len(indices_shape) > 1:
            if val_x.out_shapes[0] is not None and isinstance(
                    val_x, ONNXGraphDataNode):
                node.fluid_code.add_layer(
                    'embedding',
                    inputs=indices,
                    output=node,
                    use_fluid=True,
                    param_attr={
                        'param_attr': string(val_x.layer_name),
                        'size': val_x.out_shapes[0]
                    })
            else:
                from functools import reduce
                #indices_shape = [1,7]
                reshape_shape = reduce(lambda x, y: x * y, indices_shape)
                indices_reshape = indices.layer_name + '_shape'
                node.fluid_code.add_layer(
                    'reshape',
                    inputs=indices,
                    output=indices_reshape,
                    param_attr={'shape': [reshape_shape, ]})

                perm = list(range(len(val_x.out_shapes[0])))
                node.fluid_code.add_layer(
                    'gather',
                    inputs={'input': val_x,
                            'index': indices_reshape},
                    output=node,
                    param_attr=None)
                val_x_shape = val_x.out_shapes[0]
                reshaped_shape = []
                for i in perm:
                    reshaped_shape.append(indices_shape[i])
                for i in val_x_shape[:axis] + val_x_shape[axis + 1:]:
                    reshaped_shape.append(i)
                node.fluid_code.add_layer(
                    'reshape',
                    inputs=node,
                    output=node,
                    param_attr={'shape': reshaped_shape})
        elif axis > 0 and len(indices_shape) > 1:
            from functools import reduce
            reshape_shape = reduce(lambda x, y: x * y, indices_shape)
            indices_reshape = indices.layer_name + '_shape'
            node.fluid_code.add_layer(
                'reshape',
                inputs=indices,
                output=indices_reshape,
                param_attr={'shape': [reshape_shape, ]})

            perm = list(range(len(val_x.out_shapes[0])))
            perm = [axis] + perm[:axis] + perm[axis + 1:]
            attr_trans = {'perm': perm}
            name_trans = val_x.layer_name + '_trans'
            node.fluid_code.add_layer(
                'transpose',
                inputs=val_x,
                output=name_trans,
                param_attr=attr_trans)
            node.fluid_code.add_layer(
                'gather',
                inputs={'input': name_trans,
                        'index': indices_reshape},
                output=node,
                param_attr=None)
            node.fluid_code.add_layer(
                'transpose', inputs=node, output=node, param_attr=attr_trans)
            val_x_shape = val_x.out_shapes[0]
            reshaped_shape = []
            for i in perm:
                reshaped_shape.append(indices_shape[i])
            for i in val_x_shape[:axis] + val_x_shape[axis + 1:]:
                reshaped_shape.append(i)
            node.fluid_code.add_layer(
                'reshape',
                inputs=node,
                output=node,
                param_attr={'shape': reshaped_shape})

    @print_mapping_info
    def Range(self, node):
        val_start = self.graph.get_input_node(node, idx=0, copy=True)
        val_limit = self.graph.get_input_node(node, idx=1, copy=True)
        val_delta = self.graph.get_input_node(node, idx=2, copy=True)
        dtype = val_start.dtype
        inputs = {'start': val_start, 'end': val_limit, 'step': val_delta}
        node.fluid_code.add_layer(
            'range',
            inputs=inputs,
            output=node,
            param_attr={'dtype': string(dtype)})

    @print_mapping_info
    def Slice(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        starts, ends, axes, steps = None, None, None, None
        attr = {}
        if len(node.inputs) > 1:
            starts = self.graph.get_input_node(node, idx=1, copy=True)
            ends = self.graph.get_input_node(node, idx=2, copy=True)
            if len(node.inputs) > 3:
                axes = self.graph.get_input_node(node, idx=3, copy=True)
                axes = _const_weight_or_none(axes)
            if len(node.inputs) > 4:
                steps = self.graph.get_input_node(node, idx=4, copy=True)
                steps = _const_weight_or_none(steps)
                if steps is not None:
                    assert steps == 1, "Only support convert op:Slice, which attribute:steps == 1"
            attr = {
                "axes": axes,
                "starts": starts.layer_name,
                "ends": ends.layer_name
            }
            starts_value = _const_weight_or_none(starts)
            ends_value = _const_weight_or_none(ends)
            if starts_value is not None and ends_value is not None:
                self.omit_nodes.append(starts.layer_name)
                self.omit_nodes.append(ends.layer_name)
                ends_value = ends_value.copy()
                for idx in range(len(ends_value)):
                    if ends_value[idx] > 2**31 - 1:
                        ends_value[idx] = 2**31 - 1
                attr = {
                    "axes": axes,
                    "starts": starts_value,
                    "ends": ends_value
                }
            else:
                if starts.dtype != 'int32':
                    node.fluid_code.add_layer(
                        'cast',
                        inputs=starts,
                        output=starts,
                        param_attr={'dtype': string('int32')})
                if ends.dtype != 'int32':
                    node.fluid_code.add_layer(
                        'cast',
                        inputs=ends,
                        output=ends,
                        param_attr={'dtype': string('int32')})
        else:
            starts = node.get_attr('starts')
            ends = node.get_attr('ends')
            axes = node.get_attr('axes')
            for idx in range(len(ends)):
                if ends[idx] > 2**31 - 1:
                    ends[idx] = 2**31 - 1
            attr = {"axes": axes, "starts": starts, "ends": ends}

        node.fluid_code.add_layer(
            'slice', inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def ConstantOfShape(self, node):
        val_shape = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)

        value = node.get_attr('value')
        dtype = value.dtype
        value = value.tolist()
        assert len(value) == 1, ('given value not Scalar, shape of value > 1, '
                                 'this is not supported')
        if len(value) == 1:
            value = value[0]
            if dtype.name == 'int64':
                dtype = 'int32'
            attr = {
                'shape': val_shape.layer_name,
                'dtype': string(dtype),
                'value': value
            }
            node.fluid_code.add_layer(
                'fill_constant', inputs=None, output=node, param_attr=attr)

    @print_mapping_info
    def Split(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)

        fluid_op = 'split'
        split = node.get_attr('split')
        axis = node.get_attr('axis', 0)
        attr = {
            'num_or_sections': split,
            'dim': axis,
            'name': string(node.layer_name)
        }

        node.fluid_code.add_layer(
            'split', inputs=val_x, output=val_y, param_attr=attr)

    @print_mapping_info
    def Reshape(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_shape = self.graph.get_input_node(node, idx=1, copy=True)
        val_reshaped = self.graph.get_node(node.layer.output[0], copy=True)
        attr = {}
        shape_value = _const_weight_or_none(val_shape)
        shape_dims = len(val_shape.out_shapes[0])

        if shape_value is not None:
            node.fluid_code.add_layer(
                'reshape',
                inputs={'x': val_x},
                output=node,
                param_attr={'shape': shape_value.tolist()})
        elif val_shape.dtype == 'int64':
            val_shape_cast = val_shape.layer_name + '_cast'
            node.fluid_code.add_layer(
                'cast',
                inputs=val_shape,
                output=val_shape_cast,
                param_attr={'dtype': string('int32')})
            node.fluid_code.add_layer(
                'reshape',
                inputs=val_shape_cast,
                output=val_shape_cast,
                param_attr={'shape': val_shape.out_shapes[0]})
            node.fluid_code.add_layer(
                'reshape',
                inputs={'x': val_x,
                        'shape': val_shape_cast},
                output=node,
                param_attr=attr)
        else:
            node.fluid_code.add_layer(
                'reshape',
                inputs=val_shape,
                output=val_shape,
                param_attr={'shape': val_shape.out_shapes[0]})
            node.fluid_code.add_layer(
                'reshape',
                inputs={'x': val_x,
                        'shape': val_shape},
                output=node,
                param_attr=attr)

    @print_mapping_info
    def Cast(self, node):
        val_input = self.graph.get_input_node(node, idx=0, copy=True)
        val_output = self.graph.get_node(node.layer.output[0], copy=True)

        dtype = node.get_attr('to')
        if not isinstance(dtype, np.dtype):
            dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]

        output_dtype = val_output.dtype
        if output_dtype:
            assert dtype == output_dtype, 'dtype of to unmatches output'
        attr = {'dtype': string(dtype)}
        node.fluid_code.add_layer(
            'cast', inputs=val_input, output=node, param_attr=attr)

    @print_mapping_info
    def AveragePool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)

        auto_pad = node.get_attr('auto_pad', 'NOTSET')
        kernel_shape = node.get_attr("kernel_shape")
        poolnd = len(kernel_shape)
        strides = node.get_attr("strides")
        pad_mode = node.get_attr("pads")
        ceil_mode = bool(node.get_attr('ceil_mode', 0))
        pads = node.get_attr('pads', [0] * (poolnd * 2))
        fluid_op = 'pool{}d'.format(poolnd)
        assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'

        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x)

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            input_shape = val_x.out_shapes[0]
            pad_h = _get_same_padding(input_shape[2], kernel_shape[0],
                                      strides[0])
            pad_w = _get_same_padding(input_shape[3], kernel_shape[1],
                                      strides[1])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}

        attr = {
            "pool_size": kernel_shape,
            "pool_type": string('avg'),
            "pool_stride": strides,
            "pool_padding": paddings,
            "ceil_mode": ceil_mode,
            "exclusive": 'True',
            "name": string(node.layer_name)
        }

        node.fluid_code.add_layer(
            fluid_op, inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def Concat(self, node):
        inputs = []
        for i in range(len(node.layer.input)):
            ipt = self.graph.get_input_node(node, idx=i, copy=True)
            if isinstance(ipt, str):
                inputs.append(ipt)
            else:
                inputs.append(ipt.layer_name)
        axis = node.get_attr('axis')
        attr = {'axis': axis}
        node.fluid_code.add_layer(
            'concat', inputs=inputs, output=node, param_attr=attr)

    @print_mapping_info
    def Flatten(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axis = node.get_attr('axis', 1)
        attr = {"axis": str(axis), "name": string(node.layer_name)}
        node.fluid_code.add_layer(
            'flatten', inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def Gemm(self, node):
        val_a = self.graph.get_input_node(node, idx=0, copy=True)
        val_b = self.graph.get_input_node(node, idx=1, copy=True)
        val_c = self.graph.get_input_node(node, idx=2, copy=True)

        alpha = node.get_attr('alpha', 1.)  # optional
        beta = node.get_attr('beta', 1.)  # optional
        trans_a = bool(node.get_attr('transA', 0))  # optional
        trans_b = bool(node.get_attr('transB', 0))  # optional
        val_mm = node.layer_name + '_mm'
        matmul_inputs = {"x": val_a, "y": val_b}
        attr_matmul = {
            "transpose_x": trans_a,
            "transpose_y": trans_b,
            "alpha": alpha,
            "name": string(val_mm)
        }
        node.fluid_code.add_layer(
            'matmul',
            inputs=matmul_inputs,
            output=val_mm,
            param_attr=attr_matmul)

        if beta != 0:
            if beta == 1.:
                add_inputs = {"x": val_mm, "y": val_c}
                attr = {"name": string(node.layer_name)}
                node.fluid_code.add_layer(
                    "elementwise_add",
                    inputs=add_inputs,
                    output=node,
                    param_attr=attr)
            else:
                var_beta = node.layer_name + '_beta'
                matmul_beta_inputs = {"x": val_c, "y": var_beta}
                node.fluid_code.add_layer(
                    "Constant",
                    inputs=matmul_beta_inputs,
                    output=var_beta,
                    param_attr={'value': beta})

                add_inputs = {"x": val_mm, "y": var_beta}
                attr = {"name": string(node.layer_name)}
                node.fluid_code.add_layer(
                    "elementwise_add",
                    inputs=add_inputs,
                    output=node,
                    param_attr=attr)

    @print_mapping_info
    def Sum(self, node):
        val_inps = node.layer.input
        inputs = {
            "x": self.graph.get_input_node(
                node, idx=0, copy=True),
            "y": self.graph.get_input_node(
                node, idx=1, copy=True),
        }
        node.fluid_code.add_layer("elementwise_add", inputs=inputs, output=node)

        for idx, ipt in enumerate(val_inps[2:]):
            y = self.graph.get_input_node(node, idx=idx, copy=True)
            inputs = {
                "x": node.layer_name,
                "y": y,
            }
            node.fluid_code.add_layer(
                "elementwise_add", inputs=inputs, output=node)

    @print_mapping_info
    def MatMul(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        inputs = {"x": val_x, "y": val_y}
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer(
            "matmul", inputs=inputs, output=node, param_attr=attr)

    @print_mapping_info
    def BatchNormalization(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_scale = self.graph.get_input_node(node, idx=1, copy=True)
        val_b = self.graph.get_input_node(node, idx=2, copy=True)
        val_mean = self.graph.get_input_node(node, idx=3, copy=True)
        val_var = self.graph.get_input_node(node, idx=4, copy=True)

        self.omit_nodes.append(val_scale.layer_name)
        self.omit_nodes.append(val_b.layer_name)
        self.omit_nodes.append(val_mean.layer_name)
        self.omit_nodes.append(val_var.layer_name)

        momentum = node.get_attr('momentum', .9)
        epsilon = node.get_attr('epsilon', 1e-5)

        # Attribute: spatial is used in BatchNormalization-1,6,7
        spatial = bool(node.get_attr('spatial'))
        attr = {
            "momentum": momentum,
            "epsilon": epsilon,
            "data_layout": string('NCHW'),
            "is_test": True,
            "param_attr": string(val_scale.layer_name),
            "bias_attr": string(val_b.layer_name),
            "moving_mean_name": string(val_mean.layer_name),
            "moving_variance_name": string(val_var.layer_name),
            "use_global_stats": spatial,
            "name": string(node.layer_name)
        }
        node.fluid_code.add_layer(
            "batch_norm", inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def Transpose(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        perm = node.get_attr('perm')
        attr = {'perm': perm, "name": string(node.layer_name)}
        node.fluid_code.add_layer(
            "transpose", inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def Relu(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer(
            "relu", inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def PRelu(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_slope = self.graph.get_input_node(node, idx=1, copy=True)

        mode = 'channel'
        shape_slope = val_slope.out_shapes[0]
        if len(shape_slope) == 1:
            mode = 'all'
        elif len(shape_slope) > 2:
            mode = 'element'
        attr = {
            "param_attr": string(val_slope.layer_name),
            'mode': string(mode)
        }
        node.fluid_code.add_layer(
            "prelu", inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def Squeeze(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        attr = {'axes': axes, "name": string(node.layer_name)}
        if len(val_x.out_shapes[0]) == 1:
            node.fluid_code.add_layer(
                "cast",
                inputs=val_x,
                output=node,
                param_attr={'dtype': string(val_x.dtype)})
        else:
            node.fluid_code.add_layer(
                "squeeze", inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def Equal(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        node.fluid_code.add_layer(
            "equal",
            inputs={'x': val_x,
                    'y': val_y},
            output=node,
            param_attr=None)

    @print_mapping_info
    def Greater(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        node.fluid_code.add_layer(
            "greater_than",
            inputs={'x': val_x,
                    'y': val_y},
            output=node,
            param_attr=None)

    @print_mapping_info
    def Where(self, node):
        condition = self.graph.get_input_node(node, idx=0, copy=True)
        val_x = self.graph.get_input_node(node, idx=1, copy=True)
        val_y = self.graph.get_input_node(node, idx=2, copy=True)

        not_condition = condition.layer_name + '_not'
        node.fluid_code.add_layer(
            "logical_not",
            inputs=condition,
            output=not_condition,
            param_attr=None)
        cast_not_condition = not_condition + '_cast'
        node.fluid_code.add_layer(
            "cast",
            inputs=not_condition,
            output=cast_not_condition,
            param_attr={'dtype': string(val_x.dtype)})
        cast_condition = condition.layer_name + '_cast'
        node.fluid_code.add_layer(
            "cast",
            inputs=condition,
            output=cast_condition,
            param_attr={'dtype': string(val_x.dtype)})
        mul_val_x = val_x.layer_name + '_mul'
        node.fluid_code.add_layer(
            "elementwise_mul",
            inputs={'x': val_x,
                    'y': cast_condition},
            output=mul_val_x,
            param_attr=None)

        mul_val_y = val_y.layer_name + '_mul'
        node.fluid_code.add_layer(
            "elementwise_mul",
            inputs={'x': val_y,
                    'y': cast_not_condition},
            output=mul_val_y,
            param_attr=None)

        node.fluid_code.add_layer(
            "elementwise_add",
            inputs={'x': mul_val_x,
                    'y': mul_val_y},
            output=node,
            param_attr=None)

    @print_mapping_info
    def NonZero(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_x_dim = len(val_x.out_shapes[0])
        if val_x_dim == 1:
            node.fluid_code.add_layer("nonzero", inputs=val_x, output=val_x)
            node.fluid_code.add_layer(
                "transpose",
                inputs=val_x,
                output=node,
                param_attr={'perm': [1, 0]})
        if val_x_dim > 1:
            node.fluid_code.add_layer("nonzero", inputs=val_x, output=val_x)
            node.fluid_code.add_layer(
                "split",
                inputs=val_x,
                output=val_x,
                param_attr={'num_or_sections': 1,
                            'dim': val_x_dim})
            node.fluid_code.add_layer("concat", inputs=val_x, output=node)

    @print_mapping_info
    def Identity(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        node.fluid_code.add_layer("assign", inputs=val_x, output=node)

    @print_mapping_info
    def Tile(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_repeats = self.graph.get_input_node(node, idx=1, copy=True)
        repeats = _const_weight_or_none(val_repeats)

        if repeats is None:
            repeats = val_repeats.layer_name
        elif isinstance(repeats, int):
            repeats = [repeats]

        attr = {
            'expand_times': repeats,
            "name": string(node.layer_name),
        }
        node.fluid_code.add_layer(
            "expand", inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def MaxPool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        auto_pad = node.get_attr('auto_pad', 'NOTSET')
        assert node.get_attr(
            "dilations") is None, 'only dilations = 0 is supported'  # optional

        kernel_shape = node.get_attr("kernel_shape")
        poolnd = len(kernel_shape)
        strides = node.get_attr("strides")
        pad_mode = node.get_attr("pads")
        ceil_mode = bool(node.get_attr('ceil_mode', 0))  # optional
        pads = node.get_attr('pads', [0] * (poolnd * 2))  # optional
        fluid_op = 'pool{}d'.format(poolnd)
        assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'

        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x)

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            input_shape = val_x.out_shapes[0]
            pad_h = get_same_padding(input_shape[2], kernel_shape[0],
                                     strides[0])
            pad_w = get_same_padding(input_shape[3], kernel_shape[1],
                                     strides[1])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}

        attr = {
            "pool_size": kernel_shape,
            "pool_type": string("max"),
            "pool_stride": strides,
            "pool_padding": paddings,
            "ceil_mode": ceil_mode,
            "name": string(node.layer_name),
            "exclusive": False
        }
        node.fluid_code.add_layer(
            fluid_op, inputs=val_x, output=node, param_attr=attr)

    def _global_pool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)
        fluid_op = 'pool2d'
        pool_type = None
        if node.layer.op_type == 'GlobalMaxPool':
            pool_type = 'max'
        elif node.layer.op_type == 'GlobalAveragePool':
            pool_type = 'avg'

        attr = {
            "pool_type": string(pool_type),
            "global_pooling": True,
            "name": string(node.layer_name)
        }
        node.fluid_code.add_layer(
            fluid_op, inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def GlobalMaxPool(self, node):
        self._global_pool(node)

    @print_mapping_info
    def GlobalAveragePool(self, node):
        self._global_pool(node)

    @print_mapping_info
    def Conv(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_w = self.graph.get_input_node(node, idx=1, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)

        self.omit_nodes.append(val_w.layer_name)

        has_bias = len(node.layer.input) == 3
        if has_bias:
            val_b = self.graph.get_input_node(node, idx=2, copy=True)
            self.omit_nodes.append(val_b.layer_name)
        auto_pad = node.get_attr('auto_pad', 'NOTSET')

        kernel_shape = node.get_attr('kernel_shape')
        convnd = len(kernel_shape)
        assert 2 <= convnd <= 3, 'only conv2d and conv3d is supported'
        num_out_channels = val_w.out_shapes[0][0]
        fluid_op = 'conv{}d'.format(convnd)

        num_groups = node.get_attr('group', 1)
        strides = node.get_attr('strides', [1] * convnd)
        dilations = node.get_attr('dilations', [1] * convnd)
        assert dilations == [1] * convnd, 'only dilations == [1,1] supported,  \
            if your model have Conv with dilations > 1, please create issue on \
            https://github.com/PaddlePaddle/X2Paddle.'

        pads = node.get_attr('pads', [0] * (convnd * 2))

        input_shape = val_x.out_shapes[0]
        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x)

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            pad_h = get_same_padding(input_shape[2], kernel_shape[0],
                                     strides[0])
            pad_w = get_same_padding(input_shape[3], kernel_shape[1],
                                     strides[1])
            paddings = pad_h + pad_w

        attr = {
            "num_filters": num_out_channels,
            "filter_size": kernel_shape,
            "stride": strides,
            "padding": paddings,
            "dilation": dilations,
            "groups": num_groups,
            'param_attr': string(val_w.layer_name),
            "name": string(node.layer_name)
        }
        if has_bias:
            attr["bias_attr"] = string(val_b.layer_name)
        else:
            attr["bias_attr"] = False
        node.fluid_code.add_layer(
            fluid_op, inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def ConvTranspose(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_w = self.graph.get_input_node(node, idx=1, copy=True)
        val_b = None
        if len(node.layer.input) > 2:
            val_b = self.graph.get_input_node(node, idx=2, copy=True)
            self.omit_nodes.append(val_b.layer_name)
        self.omit_nodes.append(val_w.layer_name)

        val_y = self.graph.get_node(node.layer.output[0], copy=True)

        auto_pad = node.get_attr('auto_pad', 'NOTSET')
        out_padding = node.get_attr('output_padding', [0, 0])
        kernel_shape = node.get_attr('kernel_shape')
        assert kernel_shape, 'kernel_shape not inferred'
        convnd = len(kernel_shape)
        assert 2 <= convnd <= 3, 'only conv2d_transpose and conv3d_transpose supported'
        num_out_channels = val_w.out_shapes[0][1]
        fluid_op = 'conv{}d_transpose'.format(convnd)

        num_groups = node.get_attr('group', 1)
        strides = node.get_attr('strides', [1] * convnd)
        dilations = node.get_attr('dilations', [1] * convnd)
        output_size = node.get_attr('output_shape', [])
        pads = node.get_attr('pads', [0] * (convnd * 2))

        paddings, var_x = self._pad_if_asymmetric(node, pads, val_x)

        output_size = [0, 0]

        output_size[0] = (val_x.out_shapes[0][2] - 1
                          ) * strides[0] - 2 * paddings[0] + dilations[0] * (
                              kernel_shape[0] - 1) + 1 + out_padding[0]
        output_size[1] = (val_x.out_shapes[0][3] - 1
                          ) * strides[1] - 2 * paddings[1] + dilations[1] * (
                              kernel_shape[1] - 1) + 1 + out_padding[1]
        attr = {
            'num_filters': num_out_channels,
            'output_size': output_size or None,
            'filter_size': kernel_shape,
            'padding': paddings,
            'stride': strides,
            'dilation': dilations,
            'groups': num_groups,
            'param_attr': string(val_w.layer_name),
            'bias_attr': None if val_b is None else string(val_b.layer_name),
            'name': string(node.layer_name),
        }
        node.fluid_code.add_layer(
            fluid_op, inputs=val_x, output=node, param_attr=attr)

    @print_mapping_info
    def GRU(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_w = self.graph.get_input_node(node, idx=1, copy=True)
        val_r = self.graph.get_input_node(node, idx=2, copy=True)
        val_b = None
        val_len = None
        val_xh = None

        miss_arg_num = 0
        num_ipt = len(node.layer.input)
        if num_ipt > 3 and node.layer.input[3] != '':
            val_b = self.graph.get_input_node(node, idx=3, copy=True)
        else:
            miss_arg_num += 1
        if num_ipt > 4 and node.layer.input[4] != '':
            val_len = self.graph.get_input_node(
                node, idx=4 - miss_arg_num, copy=True)
        else:
            miss_arg_num += 1
        if num_ipt > 5 and node.layer.input[5] != '':
            val_xh = self.graph.get_input_node(
                node, idx=5 - miss_arg_num, copy=True)

        x_shape = val_x.out_shapes[0]
        assert node.get_attr('clip', None) is None, 'clipping not supported'
        assert node.get_attr('linear_before_reset',
                             0) == 0, 'only linear_before_reset = 0 supported'
        direction = node.get_attr('direction', 'forward')
        activations = node.get_attr('activations', ['Sigmoid', 'Tanh'])
        activations = [s.lower() for s in activations]
        gate_activation, candidate_activation = activations
        is_reverse = direction == 'reverse'
        w_shape = val_w.out_shapes[0]
        hidden_size = node.get_attr('hidden_size', None)
        if hidden_size is None:
            r_shape = val_r.out_shapes[0]
            if r_shape:
                hidden_size = r_shape[-1]
        if hidden_size is None:
            w_shape = val_w.out_shapes[0]
            if w_shape:
                hidden_size = w_shape[-2] // 3
        if hidden_size is None and val_b:
            b_shape = val_b.out_shapes[0]
            if b_shape:
                hidden_size = b_shape[-1] // 6
        if hidden_size is None and val_xh:
            xh_shape = val_xh.out_shapes[0]
            if xh_shape:
                hidden_size = xh_shape[-1]

        var_x0 = node.layer_name + '_x0'
        var_w0 = node.layer_name + '_w0'
        node.fluid_code.add_layer(
            'transpose',
            inputs=val_x,
            output=var_x0,
            param_attr={'perm': [1, 0, 2]})
        node.fluid_code.add_layer(
            'reshape',
            inputs=var_x0,
            output=var_x0,
            param_attr={'shape': [x_shape[1] * x_shape[0], x_shape[2]]})
        node.fluid_code.add_layer(
            'reshape',
            inputs=val_w,
            output=var_w0,
            param_attr={
                'shape': [w_shape[0] * w_shape[1], w_shape[2]],
                'name': string(var_w0)
            })
        node.fluid_code.add_layer(
            'transpose',
            inputs=var_w0,
            output=var_w0,
            param_attr={'perm': [1, 0]})
        var_fc = node.layer_name + '_fc'
        var_mm = (node.layer_name + '_mm') if val_b else var_fc

        self.omit_nodes.append(val_w.layer_name)
        node.fluid_code.add_layer(
            'mul', inputs={'x': var_x0,
                           'y': var_w0}, output=var_mm)
        if val_b:
            var_bi = node.layer_name + '_bi'
            var_bh = node.layer_name + '_bh'
            node.fluid_code.add_layer(
                'split',
                inputs=val_b,
                output=var_bi + ',' + var_bh,
                param_attr={
                    'dim': 1,
                    'num_or_sections': [hidden_size * 3, hidden_size * 3]
                })
            var_bi0 = node.layer_name + '_bi0'
            if direction == 'bidirectional':
                node.fluid_code.add_layer(
                    'reshape',
                    inputs=var_bi,
                    output=var_bi0,
                    param_attr={
                        'shape': [hidden_size * 6],
                        'name': string(var_bi0)
                    })
            else:
                node.fluid_code.add_layer(
                    'reshape',
                    inputs=var_bi,
                    output=var_bi0,
                    param_attr={
                        'shape': [hidden_size * 3],
                        'name': string(var_bi0)
                    })
            node.fluid_code.add_layer(
                'elementwise_add',
                inputs={'x': var_mm,
                        'y': var_bi0},
                output=var_fc,
                param_attr={'axis': 1})
        if direction == 'bidirectional':
            #split input for bidirection
            var_ipt_forward = node.layer_name + '_ipt_forward'
            var_ipt_backward = node.layer_name + '_ipt_backward'
            node.fluid_code.add_layer(
                'split',
                inputs=var_fc,
                output=var_ipt_forward + ',' + var_ipt_backward,
                param_attr={
                    'dim': 1,
                    'num_or_sections': [hidden_size * 3, hidden_size * 3]
                })
            #split recurrence weight tensor for bidirection
            var_r_forward = node.layer_name + '_r_forward'
            var_r_backward = node.layer_name + '_r_backward'
            val_r_value = _const_weight_or_none(val_r)
            var_r_value_forward, var_r_value_backward, _ = np.split(
                val_r_value, indices_or_sections=[1, 2], axis=0)
            var_r_value_forward = np.squeeze(var_r_value_forward, axis=0)
            [r_z, r_r, r_h] = np.split(var_r_value_forward, 3)
            var_r_value_forward = np.concatenate((
                (np.transpose(np.concatenate((r_z, r_r))).flatten(),
                 np.transpose(r_h).flatten()))).reshape(hidden_size,
                                                        hidden_size * 3)
            self.weights[var_r_forward] = var_r_value_forward

            var_r_value_backward = np.squeeze(var_r_value_backward, axis=0)
            [r_z, r_r, r_h] = np.split(var_r_value_backward, 3)
            var_r_value_backward = np.concatenate((
                (np.transpose(np.concatenate((r_z, r_r))).flatten(),
                 np.transpose(r_h).flatten()))).reshape(hidden_size,
                                                        hidden_size * 3)
            print(var_r_value_backward.shape)
            self.weights[var_r_backward] = var_r_value_backward
            node.fluid_code.add_layer(
                'split',
                inputs=val_r,
                output=var_r_forward + ',' + var_r_backward,
                param_attr={'dim': 0,
                            'num_or_sections': [1, 1]})
            node.fluid_code.add_layer(
                'squeeze',
                inputs=var_r_forward,
                output=var_r_forward,
                param_attr={'axes': [0]})
            node.fluid_code.add_layer(
                'squeeze',
                inputs=var_r_backward,
                output=var_r_backward,
                param_attr={'axes': [0]})
            if val_b:
                var_b_forward = node.layer_name + '_b_forward'
                var_b_backward = node.layer_name + '_b_backward'
                #split bias tensor of gates for bidirection
                val_b_value = _const_weight_or_none(val_b)
                var_bi_value, var_bh_value, _ = np.split(
                    val_b_value,
                    indices_or_sections=[hidden_size * 3, 2 * hidden_size * 3],
                    axis=1)
                var_bh_value_forward, var_bh_value_backward, _ = np.split(
                    var_bh_value, indices_or_sections=[1, 2], axis=0)
                self.weights[var_b_forward] = var_bh_value_forward
                self.weights[var_b_backward] = var_bh_value_backward
                node.fluid_code.add_layer(
                    'split',
                    inputs=var_bh,
                    output=var_b_forward + ',' + var_b_backward,
                    param_attr={'dim': 0,
                                'num_or_sections': [1, 1]})
            forward_attr = {
                'origin_mode': True,
                'is_reverse': False,
                'gate_activation': string(gate_activation),
                'candidate_activation': string(candidate_activation),
                'param_attr': string(var_r_forward),
                'bias_attr': string(var_b_forward) if val_b else False,
            }
            lod_attr = {
                'target_lod': [x_shape[0] * i for i in range(x_shape[1] + 1)]
            }
            node.fluid_code.add_layer(
                'lod_reset',
                inputs=var_ipt_forward,
                output=var_ipt_forward,
                param_attr=lod_attr)
            var_y_forward = node.layer_name + '_y_forward'
            node.fluid_code.add_layer(
                'dynamic_gru',
                inputs=var_ipt_forward + ',' + str(hidden_size),
                output=var_y_forward,
                param_attr=forward_attr)
            backward_attr = {
                'origin_mode': True,
                'is_reverse': True,
                'gate_activation': string(gate_activation),
                'candidate_activation': string(candidate_activation),
                'param_attr': string(var_r_backward),
                'bias_attr': string(var_b_backward) if val_b else False,
            }
            node.fluid_code.add_layer(
                'lod_reset',
                inputs=var_ipt_backward,
                output=var_ipt_backward,
                param_attr=lod_attr)
            var_y_backward = node.layer_name + '_y_backward'
            node.fluid_code.add_layer(
                'dynamic_gru',
                inputs=var_ipt_backward + ',' + str(hidden_size),
                output=var_y_backward,
                param_attr=backward_attr)
            node.fluid_code.add_layer(
                'reshape',
                inputs=var_y_forward,
                output=var_y_forward,
                param_attr={'shape': [x_shape[1], 1, x_shape[0], hidden_size]})
            node.fluid_code.add_layer(
                'reshape',
                inputs=var_y_backward,
                output=var_y_backward,
                param_attr={'shape': [x_shape[1], 1, x_shape[0], hidden_size]})
            var_y = node.layer_name + '_dynamic_gru'
            node.fluid_code.add_layer(
                'concat',
                inputs=[var_y_forward, var_y_backward],
                output=var_y,
                param_attr={'axis': 1})
            node.fluid_code.add_layer(
                'transpose',
                inputs=var_y,
                output=node.layer.output[0],
                param_attr={'perm': [2, 1, 0, 3]})
            if len(node.layer.output) > 1 and node.layer.output[1] != '':
                node.fluid_code.add_layer(
                    'transpose',
                    inputs=var_y_forward,
                    output=var_y_forward,
                    param_attr={'perm': [2, 1, 0, 3]})
                node.fluid_code.add_layer(
                    'transpose',
                    inputs=var_y_backward,
                    output=var_y_backward,
                    param_attr={'perm': [2, 1, 0, 3]})
                var_gather_index_forward = node.layer_name + '_gather_index_forward'
                node.fluid_code.add_layer(
                    'fill_constant',
                    inputs=None,
                    output=var_gather_index_forward,
                    param_attr={
                        'shape': [1],
                        'dtype': string('int32'),
                        'value': x_shape[0] - 1
                    })
                var_gather_index_backward = node.layer_name + '_gather_index_backward'
                node.fluid_code.add_layer(
                    'fill_constant',
                    inputs=None,
                    output=var_gather_index_backward,
                    param_attr={
                        'shape': [1],
                        'dtype': string('int32'),
                        'value': 0
                    })
                var_last_hidden_forward = node.layer_name + '_last_hidden_forward'
                node.fluid_code.add_layer(
                    'gather',
                    inputs=var_y_forward,
                    output=var_last_hidden_forward,
                    param_attr={'index': var_gather_index_forward})
                var_last_hidden_backward = node.layer_name + '_last_hidden_backward'
                node.fluid_code.add_layer(
                    'gather',
                    inputs=var_y_backward,
                    output=var_last_hidden_backward,
                    param_attr={'index': var_gather_index_backward})
                node.fluid_code.add_layer(
                    'concat',
                    inputs=[var_last_hidden_forward, var_last_hidden_backward],
                    output=node.layer.output[1],
                    param_attr={'axis': 1})
                node.fluid_code.add_layer(
                    'squeeze',
                    inputs=node.layer.output[1],
                    output=node.layer.output[1],
                    param_attr={'axes': [0]})
        else:
            if val_xh:
                var_xh = val_xh.layer_name + '_squeeze'
                node.fluid_code.add_layer(
                    'squeeze',
                    inputs=val_xh,
                    output=var_xh,
                    param_attr={'axes': [0]})
            if val_b:
                val_b_value = _const_weight_or_none(val_b)
                var_bi_value, var_bh_value, _ = np.split(
                    val_b_value,
                    indices_or_sections=[hidden_size * 3, 2 * hidden_size * 3],
                    axis=1)
                self.weights[var_bh] = var_bh_value
            val_r_reshaped = val_r.layer_name + '_reshaped'
            val_r_value = _const_weight_or_none(val_r)
            val_r_value = np.squeeze(val_r_value, axis=0)
            [r_z, r_r, r_h] = np.split(val_r_value, 3)
            val_r_value = np.concatenate((
                (np.transpose(np.concatenate((r_z, r_r))).flatten(),
                 np.transpose(r_h).flatten()))).reshape(hidden_size,
                                                        hidden_size * 3)
            self.omit_nodes.append(val_r.layer_name)
            self.weights[val_r_reshaped] = val_r_value
            var_y = node.layer_name + '_dynamic_gru'
            attr = {
                'origin_mode': True,
                #'h_0': var_xh if val_xh else None,
                'is_reverse': is_reverse,
                'gate_activation': string(gate_activation),
                'candidate_activation': string(candidate_activation),
                'param_attr': string(val_r_reshaped),
                'bias_attr': string(var_bh) if val_b else False,
            }
            lod_attr = {
                'target_lod': [x_shape[0] * i for i in range(x_shape[1] + 1)]
            }
            node.fluid_code.add_layer(
                'lod_reset', inputs=var_fc, output=var_fc, param_attr=lod_attr)
            node.fluid_code.add_layer(
                'dynamic_gru',
                inputs=var_fc + ',' + str(hidden_size),
                output=var_y,
                param_attr=attr)
            node.fluid_code.add_layer(
                'reshape',
                inputs=var_y,
                output=var_y,
                param_attr={'shape': [x_shape[1], x_shape[0], hidden_size]})
            node.fluid_code.add_layer(
                'transpose',
                inputs=var_y,
                output=var_y,
                param_attr={'perm': [1, 0, 2]})
            node.fluid_code.add_layer(
                'unsqueeze',
                inputs=var_y,
                output=node.layer.output[0],
                param_attr={'axes': [1]})
            num_opt = len(node.layer.output)
            if num_opt > 1 and node.layer.output[1] != '':
                var_gather_index = node.layer_name + '_gather_index'
                node.fluid_code.add_layer(
                    'fill_constant',
                    inputs=None,
                    output=var_gather_index,
                    param_attr={
                        'shape': [1],
                        'dtype': string('int32'),
                        'value': x_shape[0] - 1
                    })
                node.fluid_code.add_layer(
                    'gather',
                    inputs=var_y,
                    output=node.layer.output[1],
                    param_attr={'index': var_gather_index})

    @print_mapping_info
    def LSTM(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_w = self.graph.get_input_node(node, idx=1, copy=True)
        val_r = self.graph.get_input_node(node, idx=2, copy=True)
        val_b = None
        val_len = None
        val_xh = None

        miss_arg_num = 0
        num_ipt = len(node.layer.input)
        if num_ipt > 3 and node.layer.input[3] != '':
            val_b = self.graph.get_input_node(node, idx=3, copy=True)
        else:
            miss_arg_num += 1
        if num_ipt > 4 and node.layer.input[4] != '':
            val_len = self.graph.get_input_node(
                node, idx=4 - miss_arg_num, copy=True)
        else:
            miss_arg_num += 1
        if num_ipt > 5 and node.layer.input[5] != '':
            val_xh = self.graph.get_input_node(
                node, idx=5 - miss_arg_num, copy=True)
        else:
            miss_arg_num += 1
        if num_ipt > 6 and node.layer.input[6] != '':
            val_xc = self.graph.get_input_node(
                node, idx=6 - miss_arg_num, copy=True)
        else:
            miss_arg_num += 1
        if num_ipt > 7 and node.layer.input[7] != '':
            val_p = self.graph.get_input_node(
                node, idx=7 - miss_arg_num, copy=True)
        else:
            miss_arg_num += 1

        x_shape = val_x.out_shapes[0]
        assert node.get_attr('clip', None) is None, 'clipping not supported'
        assert node.get_attr('input_forget',
                             0) == 0, 'only input_forget = 0 supported'
        direction = node.get_attr('direction', 'forward')
        activations = node.get_attr('activations', ['Sigmoid', 'Tanh', 'Tanh'])
        activations = [s.lower() for s in activations]
        gate_activation, cell_activation, candidate_activation = activations
        is_reverse = direction == 'reverse'
        w_shape = val_w.out_shapes[0]
        hidden_size = node.get_attr('hidden_size', None)
        if hidden_size is None:
            r_shape = val_r.out_shapes[0]
            if r_shape:
                hidden_size = r_shape[-1]
        if hidden_size is None:
            w_shape = val_w.out_shapes[0]
            if w_shape:
                hidden_size = w_shape[-2] // 4
        if hidden_size is None and val_b:
            b_shape = val_b.out_shapes[0]
            if b_shape:
                hidden_size = b_shape[-1] // 8
        if hidden_size is None and val_xh:
            xh_shape = val_xh.out_shapes[0]
            if xh_shape:
                hidden_size = xh_shape[-1]

        var_x0 = node.layer_name + '_x_reshape'
        var_w0 = node.layer_name + '_w_reshape'
        node.fluid_code.add_layer(
            'transpose',
            inputs=val_x,
            output=var_x0,
            param_attr={'perm': [1, 0, 2]})
        node.fluid_code.add_layer(
            'reshape',
            inputs=var_x0,
            output=var_x0,
            param_attr={'shape': [x_shape[1] * x_shape[0], x_shape[2]]})
        node.fluid_code.add_layer(
            'reshape',
            inputs=val_w,
            output=var_w0,
            param_attr={
                'shape': [w_shape[1] * w_shape[0], w_shape[2]],
                #'shape': [w_shape[2], w_shape[1] * w_shape[0]],
                'name': string(var_w0)
            })
        node.fluid_code.add_layer(
            'transpose',
            inputs=var_w0,
            output=var_w0,
            param_attr={'perm': [1, 0]})
        var_fc = node.layer_name + '_fc'
        var_mm = (node.layer_name + '_mm') if val_b else var_fc

        self.omit_nodes.append(val_w.layer_name)
        node.fluid_code.add_layer(
            'mul', inputs={'x': var_x0,
                           'y': var_w0}, output=var_mm)
        if val_b:
            var_bi = node.layer_name + '_bi'
            var_bh = node.layer_name + '_bh'
            node.fluid_code.add_layer(
                'split',
                inputs=val_b,
                output=var_bi + ',' + var_bh,
                param_attr={
                    'dim': 1,
                    'num_or_sections': [hidden_size * 4, hidden_size * 4]
                })
            var_bi0 = node.layer_name + '_bi0'
            if direction == 'bidirectional':
                node.fluid_code.add_layer(
                    'reshape',
                    inputs=var_bi,
                    output=var_bi0,
                    param_attr={
                        'shape': [hidden_size * 8],
                        'name': string(var_bi0)
                    })
            else:
                node.fluid_code.add_layer(
                    'reshape',
                    inputs=var_bi,
                    output=var_bi0,
                    param_attr={
                        'shape': [hidden_size * 4],
                        'name': string(var_bi0)
                    })
            node.fluid_code.add_layer(
                'elementwise_add',
                inputs={'x': var_mm,
                        'y': var_bi0},
                output=var_fc,
                param_attr={'axis': 1})
        if direction == 'bidirectional':
            #split input for bidirection
            var_ipt_forward = node.layer_name + '_ipt_forward'
            var_ipt_backward = node.layer_name + '_ipt_backward'
            node.fluid_code.add_layer(
                'split',
                inputs=var_fc,
                output=var_ipt_forward + ',' + var_ipt_backward,
                param_attr={
                    'dim': 1,
                    'num_or_sections': [hidden_size * 4, hidden_size * 4]
                })
            #split recurrence weight tensor for bidirection
            var_r_forward = node.layer_name + '_r_forward'
            var_r_backward = node.layer_name + '_r_backward'
            val_r_value = _const_weight_or_none(val_r)
            var_r_value_forward, var_r_value_backward, _ = np.split(
                val_r_value, indices_or_sections=[1, 2], axis=0)
            var_r_value_forward = np.squeeze(var_r_value_forward, axis=0)
            [r_z, r_r, r_o, r_h] = np.split(var_r_value_forward, 4)
            var_r_value_forward = np.concatenate((
                (np.transpose(np.concatenate((r_z, r_r))).flatten(),
                 np.transpose(np.concatenate((r_o, r_h))).flatten()))).reshape(
                     hidden_size, hidden_size * 4)
            self.weights[var_r_forward] = var_r_value_forward

            var_r_value_backward = np.squeeze(var_r_value_backward, axis=0)
            [r_z, r_r, r_o, r_h] = np.split(var_r_value_backward, 4)
            var_r_value_backward = np.concatenate((
                (np.transpose(np.concatenate((r_z, r_r))).flatten(),
                 np.transpose(np.concatenate((r_o, r_h))).flatten()))).reshape(
                     hidden_size, hidden_size * 4)
            self.weights[var_r_backward] = var_r_value_backward
            node.fluid_code.add_layer(
                'split',
                inputs=val_r,
                output=var_r_forward + ',' + var_r_backward,
                param_attr={'dim': 0,
                            'num_or_sections': [1, 1]})
            node.fluid_code.add_layer(
                'squeeze',
                inputs=var_r_forward,
                output=var_r_forward,
                param_attr={'axes': [0]})
            node.fluid_code.add_layer(
                'squeeze',
                inputs=var_r_backward,
                output=var_r_backward,
                param_attr={'axes': [0]})
            if val_b:
                var_b_forward = node.layer_name + '_b_forward'
                var_b_backward = node.layer_name + '_b_backward'
                #split bias tensor of gates for bidirection
                val_b_value = _const_weight_or_none(val_b)
                var_bi_value, var_bh_value, _ = np.split(
                    val_b_value,
                    indices_or_sections=[hidden_size * 4, 2 * hidden_size * 4],
                    axis=1)
                var_bh_value_forward, var_bh_value_backward, _ = np.split(
                    var_bh_value, indices_or_sections=[1, 2], axis=0)
                self.weights[var_b_forward] = var_bh_value_forward
                self.weights[var_b_backward] = var_bh_value_backward
                node.fluid_code.add_layer(
                    'split',
                    inputs=var_bh,
                    output=var_b_forward + ',' + var_b_backward,
                    param_attr={'dim': 0,
                                'num_or_sections': [1, 1]})
            forward_attr = {
                'is_reverse': False,
                'gate_activation': string(gate_activation),
                'cell_activation': string(cell_activation),
                'candidate_activation': string(candidate_activation),
                'param_attr': string(var_r_forward),
                'bias_attr': string(var_b_forward) if val_b else False,
            }
            lod_attr = {
                'target_lod': [x_shape[0] * i for i in range(x_shape[1] + 1)]
            }
            node.fluid_code.add_layer(
                'lod_reset',
                inputs=var_ipt_forward,
                output=var_ipt_forward,
                param_attr=lod_attr)
            var_y_forward = node.layer_name + '_y_forward'
            node.fluid_code.add_layer(
                'dynamic_gru',
                inputs=var_ipt_forward + ',' + str(hidden_size),
                output=var_y_forward,
                param_attr=forward_attr)
            backward_attr = {
                'is_reverse': True,
                'gate_activation': string(gate_activation),
                'cell_activation': string(cell_activation),
                'candidate_activation': string(candidate_activation),
                'param_attr': string(var_r_backward),
                'bias_attr': string(var_b_backward) if val_b else False,
            }
            node.fluid_code.add_layer(
                'lod_reset',
                inputs=var_ipt_backward,
                output=var_ipt_backward,
                param_attr=lod_attr)
            var_y_backward = node.layer_name + '_y_backward'
            node.fluid_code.add_layer(
                'dynamic_gru',
                inputs=var_ipt_backward + ',' + str(hidden_size),
                output=var_y_backward,
                param_attr=backward_attr)
            node.fluid_code.add_layer(
                'reshape',
                inputs=var_y_forward,
                output=var_y_forward,
                param_attr={'shape': [x_shape[1], 1, x_shape[0], hidden_size]})
            node.fluid_code.add_layer(
                'reshape',
                inputs=var_y_backward,
                output=var_y_backward,
                param_attr={'shape': [x_shape[1], 1, x_shape[0], hidden_size]})
            var_y = node.layer_name + '_dynamic_gru'
            node.fluid_code.add_layer(
                'concat',
                inputs=[var_y_forward, var_y_backward],
                output=var_y,
                param_attr={'axis': 1})
            node.fluid_code.add_layer(
                'transpose',
                inputs=var_y,
                output=node.layer.output[0],
                param_attr={'perm': [2, 1, 0, 3]})
            if len(node.layer.output) > 1 and node.layer.output[1] != '':
                node.fluid_code.add_layer(
                    'transpose',
                    inputs=var_y_forward,
                    output=var_y_forward,
                    param_attr={'perm': [2, 1, 0, 3]})
                node.fluid_code.add_layer(
                    'transpose',
                    inputs=var_y_backward,
                    output=var_y_backward,
                    param_attr={'perm': [2, 1, 0, 3]})
                var_gather_index_forward = node.layer_name + '_gather_index_forward'
                node.fluid_code.add_layer(
                    'fill_constant',
                    inputs=None,
                    output=var_gather_index_forward,
                    param_attr={
                        'shape': [1],
                        'dtype': string('int32'),
                        'value': x_shape[0] - 1
                    })
                var_gather_index_backward = node.layer_name + '_gather_index_backward'
                node.fluid_code.add_layer(
                    'fill_constant',
                    inputs=None,
                    output=var_gather_index_backward,
                    param_attr={
                        'shape': [1],
                        'dtype': string('int32'),
                        'value': 0
                    })
                var_last_hidden_forward = node.layer_name + '_last_hidden_forward'
                node.fluid_code.add_layer(
                    'gather',
                    inputs=var_y_forward,
                    output=var_last_hidden_forward,
                    param_attr={'index': var_gather_index_forward})
                var_last_hidden_backward = node.layer_name + '_last_hidden_backward'
                node.fluid_code.add_layer(
                    'gather',
                    inputs=var_y_backward,
                    output=var_last_hidden_backward,
                    param_attr={'index': var_gather_index_backward})
                node.fluid_code.add_layer(
                    'concat',
                    inputs=[var_last_hidden_forward, var_last_hidden_backward],
                    output=node.layer.output[1],
                    param_attr={'axis': 1})
                node.fluid_code.add_layer(
                    'squeeze',
                    inputs=node.layer.output[1],
                    output=node.layer.output[1],
                    param_attr={'axes': [0]})
        else:
            if val_xh:
                var_xh = val_xh.layer_name + '_squeeze'
                node.fluid_code.add_layer(
                    'squeeze',
                    inputs=val_xh,
                    output=var_xh,
                    param_attr={'axes': [0]})
            if val_b:
                val_b_value = _const_weight_or_none(val_b)
                var_bi_value, var_bh_value, _ = np.split(
                    val_b_value,
                    indices_or_sections=[hidden_size * 4, 2 * hidden_size * 4],
                    axis=1)
                self.weights[var_bh] = var_bh_value
            val_r_reshaped = val_r.layer_name + '_reshaped'
            val_r_value = _const_weight_or_none(val_r)
            val_r_value = np.squeeze(val_r_value, axis=0)
            #val_r_value = np.transpose(val_r_value, axes=[1,0])
            val_r_value = np.reshape(
                val_r_value, [val_r_value.shape[1], val_r_value.shape[0]],
                order='A')
            #[r_z, r_r, r_o, r_h] = np.split(val_r_value, 4)
            #val_r_value = np.concatenate(
            #    (np.transpose(np.concatenate((r_z, r_r))).flatten(),
            #     np.transpose(np.concatenate((r_o, r_h))).flatten())).reshape(
            #         hidden_size, hidden_size * 4)
            self.omit_nodes.append(val_r.layer_name)
            self.weights[val_r_reshaped] = val_r_value
            var_y = node.layer_name + '_dynamic_lstm'
            var_cell = node.layer_name + '_dynamic_lstm_cell'
            attr = {
                #'h_0': var_xh if val_xh else None,
                'is_reverse': is_reverse,
                'gate_activation': string(gate_activation),
                'cell_activation': string(cell_activation),
                'candidate_activation': string(candidate_activation),
                'param_attr': string(val_r_reshaped),
                'bias_attr': string(var_bh) if val_b else False,
                'use_peepholes': False
            }
            lod_attr = {
                'target_lod': [x_shape[0] * i for i in range(x_shape[1] + 1)]
            }
            node.fluid_code.add_layer(
                'lod_reset', inputs=var_fc, output=var_fc, param_attr=lod_attr)
            node.fluid_code.add_layer(
                'dynamic_lstm',
                inputs=var_fc + ',' + str(hidden_size * 4),
                output=var_y + ',' + var_cell,
                param_attr=attr)
            node.fluid_code.add_layer(
                'reshape',
                inputs=var_y,
                output=var_y,
                param_attr={'shape': [x_shape[1], x_shape[0], hidden_size]})
            node.fluid_code.add_layer(
                'transpose',
                inputs=var_y,
                output=var_y,
                param_attr={'perm': [1, 0, 2]})
            node.fluid_code.add_layer(
                'unsqueeze',
                inputs=var_y,
                output=node.layer.output[0],
                param_attr={'axes': [1]})

            node.fluid_code.add_layer(
                'reshape',
                inputs=var_cell,
                output=var_cell,
                param_attr={'shape': [x_shape[1], x_shape[0], hidden_size]})
            node.fluid_code.add_layer(
                'transpose',
                inputs=var_cell,
                output=var_cell,
                param_attr={'perm': [1, 0, 2]})

            num_opt = len(node.layer.output)
            if num_opt > 1 and node.layer.output[1] != '':
                var_hidden_gather_index = node.layer_name + '_hidden_gather_index'
                node.fluid_code.add_layer(
                    'fill_constant',
                    inputs=None,
                    output=var_hidden_gather_index,
                    param_attr={
                        'shape': [1],
                        'dtype': string('int32'),
                        'value': x_shape[0] - 1
                    })
                node.fluid_code.add_layer(
                    'gather',
                    inputs=var_y,
                    output=node.layer.output[1],
                    param_attr={'index': var_hidden_gather_index})
            if num_opt > 2 and node.layer.output[2] != '':
                var_cell_gather_index = node.layer_name + '_cell_gather_index'
                node.fluid_code.add_layer(
                    'fill_constant',
                    inputs=None,
                    output=var_cell_gather_index,
                    param_attr={
                        'shape': [1],
                        'dtype': string('int32'),
                        'value': x_shape[0] - 1
                    })
                node.fluid_code.add_layer(
                    'gather',
                    inputs=var_cell,
                    output=node.layer.output[2],
                    param_attr={'index': var_cell_gather_index})
