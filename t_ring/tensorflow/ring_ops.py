# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2017 NEWPLAN Tsinghua University.
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
# =============================================================================
"""Inter-process communication using TCP or RDMA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import re
import sysconfig
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader


def _get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def _load_library(name, op_list=None):
    """Loads a .so file containing the specified operators.

    Args:
      name: The name of the .so file to load.
      op_list: A list of names of operators that the library should have. If None
          then the .so file's contents will not be verified.

    Raises:
      NameError if one of the required ops is missing.
      NotFoundError if were not able to load .so file.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    for expected_op in (op_list or []):
        for lib_op in library.OP_LIST.op:
            if lib_op.name == expected_op:
                break
        else:
            raise NameError(
                'Could not find operator %s in dynamic library %s' %
                (expected_op, name))
    return library


def _load_ctypes_dll(name):
    filename = resource_loader.get_path_to_datafile(name)
    return ctypes.CDLL(filename, mode=ctypes.RTLD_GLOBAL)


RING_LIB = _load_library('ring_lib' + _get_ext_suffix(),
                        ['RingAllreduce','RingBroadcast'])
#ddd


RING_LIB_CTYPES = _load_ctypes_dll('ring_lib' + _get_ext_suffix())


def init():
    """A function which initializes Ring.
    """
    return RING_LIB_CTYPES.ring_tensorflow_init()


def size():
    """A function which returns the number of Ring processes.

    Returns:
      An integer scalar containing the number of Ring processes.
    """
    size = RING_LIB_CTYPES.ring_tensorflow_size()
    if size == -1:
        raise ValueError(
            'Ring has not been initialized; use ring.tensorflow.init().')
    return size


def rank():
    """A function which returns the Ring rank of the calling process.

    Returns:
      An integer scalar with the Ring rank of the calling process.
    """
    rank = RING_LIB_CTYPES.ring_tensorflow_rank()
    if rank == -1:
        raise ValueError(
            'Ring has not been initialized; use ring.tensorflow.init().')
    return rank


def local_rank():
    """A function which returns the local Ring rank of the calling process, within the
    node that it is running on. For example, if there are seven processes running
    on a node, their local ranks will be zero through six, inclusive.

    Returns:
      An integer scalar with the local Ring rank of the calling process.
    """
    local_rank = RING_LIB_CTYPES.ring_tensorflow_local_rank()
    if local_rank == -1:
        raise ValueError(
            'Ring has not been initialized; use ring.tensorflow.init().')
    return local_rank


def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)


def _allreduce(tensor, name=None):
    """An op which sums an input tensor over all the Ring processes.

    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Ring processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.
    """
    if name is None:
        name = 'RingAllreduce_%s' % _normalize_name(tensor.name)
    return RING_LIB.ring_allreduce(tensor, name=name)


ops.NotDifferentiable('RingAllreduce')


def allgather(tensor, name=None):
    """An op which concatenates the input tensor with the same input tensor on
    all other Ring processes.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.

    Returns:
      A tensor of the same type as `tensor`, concatenated on dimension zero
      across all processes. The shape is identical to the input shape, except for
      the first dimension, which may be greater and is the sum of all first
      dimensions of the tensors in different Ring processes.
    """
    if name is None:
        name = 'RingAllgather_%s' % _normalize_name(tensor.name)
    return RING_LIB.ring_allgather(tensor, name=name)


ops.NotDifferentiable('RingAllgather')


def broadcast(tensor, root_rank, name=None):
    """An op which broadcasts the input tensor on root rank to the same input tensor
    on all other Ring processes.

    The broadcast operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Ring processes for a given name. The broadcast
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, with the value broadcasted
      from root rank.
    """
    if name is None:
        name = 'RingBroadcast_%s' % _normalize_name(tensor.name)
    return RING_LIB.ring_broadcast(tensor, name=name, root_rank=root_rank)


ops.NotDifferentiable('RingBroadcast')
