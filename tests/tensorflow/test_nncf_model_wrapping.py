#  Copyright (c) 2023 Intel Corporation
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from nncf.common.graph import NNCFGraph
from nncf.common.nncf_model import wrap_model
from tests.tensorflow.helpers import get_basic_two_conv_test_model
import tensorflow as tf

def test_nncf_model_wrapping():
    orig_model = get_basic_two_conv_test_model()
    nncf_model = wrap_model(orig_model)
    assert isinstance(nncf_model.nncf.graph, NNCFGraph)

    old_graph_id = id(nncf_model.nncf.graph)
    new_graph = nncf_model.nncf.rebuild_graph()
    assert new_graph is nncf_model.nncf.graph
    assert id(new_graph) != old_graph_id

    released = nncf_model.nncf.release()
    assert isinstance(released, tf.Module)
    assert not hasattr(released, "nncf")