# Copyright 2020 The PGDL Competition organizers.
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

# Examples of organizer-provided metrics.
# Main contributor: Yiding Jiang, July 2020-October 2020

import numpy as np
import scipy as sp
from mutual_information import conditional_mutual_information


def mutual_information(prediction, reference):
  return conditional_mutual_information(prediction, reference)

