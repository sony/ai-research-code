# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os
import nnabla as nn
from nnabla.parameter import get_parameter_or_create
from nnabla.ext_utils import get_extension_context
from args import get_train_args
from data import load_datasources
from utils import get_statistics

# To improve load performance
os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = str(1)

parser, args = get_train_args()

# Get context.
ctx = get_extension_context(args.context, device_id=args.device_id)
nn.set_default_context(ctx)

train_source, _, _ = load_datasources(parser, args)
input_mean, input_scale = get_statistics(args, train_source)

# Save stats as nnabla parameters
get_parameter_or_create('input_mean', initializer=input_mean)
get_parameter_or_create('input_scale', initializer=input_scale)
nn.save_parameters(args.stats)