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

from x2paddle.op_mapper.onnx2paddle.opset9.opset import OpSet9
import numpy as np
import logging as _logging

_logger = _logging.getLogger(__name__)


class OpSet10(OpSet9):
    def __init__(self, decoder):
        super(OpSet10, self).__init__(decoder)
