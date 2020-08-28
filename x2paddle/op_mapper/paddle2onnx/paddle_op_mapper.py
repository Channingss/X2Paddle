#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import math
import sys
import x2paddle
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from x2paddle.op_mapper.paddle2onnx.opset9.opset import OpSet9
from x2paddle.op_mapper.paddle2onnx.opset10.opset import OpSet10
from x2paddle.op_mapper.paddle2onnx.opset11.opset import OpSet11
from paddle.fluid.executor import  scope_guard
from paddle.fluid.dygraph.base import program_desc_tracing_guard, switch_to_static_graph

class PaddleOpMapper(object):
    def __init__(self):
        self.paddle_onnx_dtype_map = {
            core.VarDesc.VarType.FP32: onnx_pb.TensorProto.FLOAT,
            core.VarDesc.VarType.FP64: onnx_pb.TensorProto.DOUBLE,
            core.VarDesc.VarType.INT32: onnx_pb.TensorProto.INT32,
            core.VarDesc.VarType.INT16: onnx_pb.TensorProto.INT16,
            core.VarDesc.VarType.INT16: onnx_pb.TensorProto.UINT16,
            core.VarDesc.VarType.INT64: onnx_pb.TensorProto.INT64,
            core.VarDesc.VarType.BOOL: onnx_pb.TensorProto.BOOL
        }
        self.support_opsets = [9, 10, 11]
        self.default_opset = 10
        self.name_counter = dict()
        self.op_set = None

    def convert_weights(self, parameters):
        nodes = list()
        for param in parameters:
            if param.name.endswith('feed') or param.name.endswith('fetch'):
                continue
            if not param.persistable:
                continue
            #print(param.data)
            #break
            weight = np.array(param.value().get_tensor())
            tensor = helper.make_tensor(
                name=param.name,
                dims=param.shape,
                data_type=self.paddle_onnx_dtype_map[param.dtype],
                vals=weight.flatten().tolist())
            node = helper.make_node(
                'Constant', inputs=[], outputs=[param.name], value=tensor)
            nodes.append(node)
        return nodes

    @switch_to_static_graph
    def convert(self, concrete_program, save_dir, scope=None, opset_version=10):
        program = concrete_program.main_program.clone()
        #for param in concrete_program.parameters:
        #    print(dir(param))
        #    break
        self.op_set = self.create_opset(opset_version)
        weight_nodes = self.convert_weights(concrete_program.parameters)
        op_nodes = list()
        input_nodes = list()
        output_nodes = list()
        unsupported_ops = set()

        print("Translating PaddlePaddle to ONNX...\n")
        for block in program.blocks:
            for feed in concrete_program.inputs:
                if isinstance(feed, fluid.Variable):
                    node = getattr(self.op_set, 'feed')(feed, block)
                    input_nodes.append(node)
            for fetch in concrete_program.outputs:
                node = getattr(self.op_set, 'fetch')(fetch, block)
                output_nodes.append(node)
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                sys.stdout.write("\rTotal:{}, Current:{} : {} ".format(
                    len(block.ops), i + 1, op.type))
                sys.stdout.flush()
                if not hasattr(self.op_set, op.type):
                    unsupported_ops.add(op.type)
                    continue
                if len(unsupported_ops) > 0:
                    continue
                node = getattr(self.op_set, op.type)(op, block)
                if op.type == 'feed':
                    print(node.name)
                    input_nodes.append(node)
                elif op.type == 'fetch':
                    output_nodes.append(node)
                else:
                    if isinstance(node, list):
                        op_nodes = op_nodes + node
                    else:
                        op_nodes.append(node)

        if len(unsupported_ops) > 0:
            print("\nThere's {} ops are not supported yet".format(
                len(unsupported_ops)))
            for op in unsupported_ops:
                print("=========== {} ===========".format(op))
            return

        graph = helper.make_graph(
            nodes=weight_nodes + op_nodes,
            name='onnx_model_from_paddle',
            initializer=[],
            inputs=input_nodes,
            outputs=output_nodes)
        opset_imports = [helper.make_opsetid("", opset_version)]
        model = helper.make_model(
            graph, producer_name='X2Paddle', opset_imports=opset_imports)
        onnx.checker.check_model(model)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'x2paddle_model.onnx'), 'wb') as f:
            f.write(model.SerializeToString())
        print("\nTranslated model saved in {}".format(
            os.path.join(save_dir, 'x2paddle_model.onnx')))

    def create_opset(self, opset_version=10):
        run_opset = self.default_opset
        opset = ''
        if opset_version in self.support_opsets:
            run_opset = opset_version
        else:
            for support_opset_version in self.support_opsets:
                if support_opset_version < opset_version:
                    run_opset = support_opset_version
                else:
                    break
        print(
            'Now, onnx2paddle support convert onnx model opset_verison {},'
            'opset_verison of your onnx model is {}, automatically treated as op_set: {}.'
            .format(self.support_opsets, opset_version, run_opset))
        opset = 'OpSet' + str(run_opset)
        return eval(opset)()
