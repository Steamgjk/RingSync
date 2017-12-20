/*==============================================================================
# Copyright 2017 NEWPLAN, Tsinghua University. All Rights Reserved.
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
# ==============================================================================
*/

#ifndef __BCUBE__MESSAGE__
#define __BCUBE__MESSAGE__

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <unordered_map>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


#define EIGEN_USE_THREADS

#if HAVE_CUDA
#include "tensorflow/stream_executor/stream.h"
#include <cuda_runtime.h>
#endif
using namespace tensorflow;

#define __have__tensorflow__

#ifdef __have__tensorflow__
typedef std::function<void(const Status&)> Status_callback;
#endif

// Use void pointer for ready event if CUDA is not present to avoid linking
// error.
#if HAVE_CUDA
#define GPU_EVENT_IF_CUDA perftools::gputools::Event*
#else
#define GPU_EVENT_IF_CUDA void*
#endif

/*
typedef enum
{
	T_INIT8 = 0, T_INT32 = 1, T_INIT64 = 2, T_FLOAT32 = 3, T_FLOAT64 = 4
} TENSOR_TYPE;
*/

typedef enum
{
	T_VOID    = 0,  T_BOOL    = 1,
	T_UINIT8  = 2,  T_INIT8   = 3,
	T_UINT16  = 4,  T_INT16   = 5,
	T_UINT32  = 6,  T_INT32   = 7,
	T_UINT64  = 8,  T_INT64   = 9,
	T_FLOAT32 = 10, T_FLOAT64 = 11
} TENSOR_TYPE;

typedef TENSOR_TYPE BCUBE_TYPE;


typedef enum
{
	OPS_ERROR = 0, ALLREDUCE = 1,
	ALLGATHER = 2, BROADCAST = 3,
	OPS_HELLO = 4
} TENSOR_OPS;

typedef struct
{
	int rank;/*which rank sended*/
	int start_pos;/*start position in a flatten tensor*/
	int nums;/*tensor element nums*/
	int msg_length;
	int name_len;/*tensor name length*/
	TENSOR_TYPE tensor_type;/*tensor type*/
	TENSOR_OPS t_ops;/*current tensor ops*/
	char data[1];/*data store tensor_name and tensor_data,offset by name_length and tensor_len*/
} msg_struct;


typedef int Tensor_Shape;

typedef struct
{
	Tensor_Shape tensor_shape;
	void* tensor_ptr;
} Tensor_Info;

typedef struct
{
	std::vector<bool> step;/*use less*/
	std::vector<int> block_in_step;/*use less*/


	std::string tensor_name;/*store tensor name*/
	TENSOR_TYPE tensor_type;/*current tensor type*/
	std::size_t available_nums;/*available nums, for alignment in reduce.*/
	std::size_t block_size;/*element number in each block*/

	void* tensor_data = nullptr; /*store tensor data*/
	int tensor_size;

#ifdef __have__tensorflow__
	OpKernelContext* context;/*context for tensor*/
	Tensor tensor;/*Input tensor.*/
	Tensor* output;/* Pre-allocated output tensor. */
	int device;
	GPU_EVENT_IF_CUDA ready_event;// Event indicating that data is ready.
	Status_callback callback;
#endif

	/*index as order=rand,tensorshape describe a tensor like size, void* for tensor ptr.*/
	std::vector<Tensor_Info> gather_tensor;
	TENSOR_OPS tensor_ops;/*tensor operation like allreduce,allgather,broadcast...*/
	std::vector<int64_t> tensor_shape;
} tensor_table_entry;


typedef struct
{
	std::string tensor_name;/*tensor name*/
	std::size_t tensor_nums;/*number of element*/
	std::size_t start_position;/*start position*/
	TENSOR_TYPE tensor_type;/*tensor type, int, char or other*/
	TENSOR_OPS tensor_ops;/*tensor ops allreduce,allgather,broadcast*/
	char* receive_ptr = nullptr; /*point to the memory access for tensor data*/
	std::vector<Tensor_Info> gather_ptr;
} received_tensor_entry;

typedef std::unordered_map<std::string, tensor_table_entry> Tensor_table;

class tensor_msg
{
public:
	tensor_msg() {}; /*nothing to do*/
	~tensor_msg() {};

	static void encode(tensor_table_entry&, void**, int, int, int*);/*encode to send*/
	static void decode(received_tensor_entry&, void* );/*decode msg to tensor entery*/
};
received_tensor_entry msg_deocde(void* msg);
#endif // __BCUBE__MESSAGE__
