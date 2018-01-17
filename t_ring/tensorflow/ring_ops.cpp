/*===============================================================================
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2017 NEWPLAN Tsinghua University.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

/*
* Allreduce, Allgather and Broadcast Ops for TensorFlow.
*
* TensorFlow natively provides inter-device communication through send and
* receive ops and inter-node communication through Distributed TensorFlow,
* based on the same send and receive abstractions. These end up being
* insufficient for synchronous data-parallel training on HPC clusters where
* Infiniband or other high-speed interconnects are available.  This module
* implements BCUBE ops for allgather, allreduce and broadcast, which do
* optimized gathers, reductions and broadcasts and can take advantage of topology
* of bcube.
*
* The primary logic of the allreduce, allgather and broadcast are in TCP and
* RDMA implementations. The background thread which facilitates BCUBE operations
* is run in BackgroundThreadLoop(). The provided ops are:
*      - RingAllreduce:
*          Perform an allreduce on a Tensor, returning the sum
*          across all BCUBE nodes in the global topology.
*      - RingAllgather:
*          Perform an allgather on a Tensor, returning the concatenation of
*          the tensor on the first dimension across allBCUBE nodes in the
*          global Topology.
*      - RingBroadcast:
*          Perform a broadcast on a Tensor, broadcasting Tensor
*          value from root rank(RANK_0) to all other ranks.
*
* Additionally, this library provides C APIs to initialize Ring and query
* rank, and world size.  These are used in Python directly through ctypes.
*/
#include "ring_ops.h"
#include <iostream>
#include <atomic>
#include <thread>
#include <string>
#include <assert.h>
#include <cstring>

using namespace tensorflow;
//bcube_global_struct bcube_gs;
MyRing* mring = NULL;

class MyhrdthryhdrhendClass
{
public:
	MyhrdthryhdrhendClass()
	{
	}
	~MyhrdthryhdrhendClass()
	{
		if (mring)
		{
			printf("Come\n");
			mring->ShutDown();
			delete mring;
		}

	}

};
MyhrdthryhdrhendClass ed;

namespace t_ring
{
namespace tensorflow
{
namespace
{

#if HAVE_CUDA
inline bool check_cuda(TensorRingStruct& trs, std::string op_name, cudaError_t result)
{
	//printf("In Check Cuda\n");
	if (result != cudaSuccess)
	{
//#ifdef __BCUBE_DEBUG__
		printf("%s failed: error in tensor:%s  result = %d  %s\n", op_name.c_str(), trs.tensor_name.c_str(), result, cudaGetErrorString(result) );
//#endif
		trs.callback(errors::Unknown(op_name, " failed: ", cudaGetErrorString(result)));
		//printf("Check Cuda 7\n");
		return false;
	}
	return true;
}

#endif


static  int TYPE_SIZE[] =
{
	4,				sizeof(bool),
	sizeof(uint8_t), sizeof(int8_t),
	sizeof(uint16_t), sizeof(int16_t),
	sizeof(uint32_t), sizeof(int32_t),
	sizeof(uint64_t), sizeof(int64_t),
	sizeof(float_t), sizeof(double_t)
};
// Convert a TensorFlow DataType to our MPIDataType.
Status DataTypeToRingType(DataType tf_dtype, RING_TYPE* ring_dtype)
{
	switch (tf_dtype)
	{
	case DT_UINT8:
	{
		*ring_dtype = T_UINIT8;
		return Status::OK();
	}
	case DT_INT8:
	{
		*ring_dtype = T_INIT8;
		return Status::OK();
	}
	case DT_UINT16:
	{
		*ring_dtype = T_UINT16;
		return Status::OK();
	}
	case DT_INT16:
	{
		*ring_dtype = T_INT16;
		return Status::OK();
	}
	case DT_INT32:
	{
		*ring_dtype = T_INT32;
		return Status::OK();
	}
	case DT_INT64:
	{
		*ring_dtype = T_INT64;
		return Status::OK();
	}
	case DT_FLOAT:
	{
		*ring_dtype = T_FLOAT32;
		return Status::OK();
	}
	case DT_DOUBLE:
	{
		*ring_dtype = T_FLOAT64;
		return Status::OK();
	}
	case DT_BOOL:
	{
		*ring_dtype = T_BOOL;
		return Status::OK();
	}
	default:
	{
		return errors::Internal("Invalid tensor type.");
	}
	}
}

// Check that Ring is initialized.
Status CheckInitialized()
{
	if (NULL == mring || (!mring->get2LeftConnected()) || (!mring->get2RightConnected()) )
	{
		return errors::FailedPrecondition(
		           "Ring has not been initialized; use bcube.tensorflow.init().");
	}

	return Status::OK();
}

// ring must be initialized and the background thread must be running before this function is called.
void ring_allreduce_queue(OpKernelContext* context, const Tensor& tensor,
                          Tensor* output, GPU_EVENT_IF_CUDA ready_event,
                          const std::string name, const int device,
                          StatusCallback callback)
{

	RING_TYPE dtype;
	Status status = DataTypeToRingType(tensor.dtype(), &dtype);
	if (!status.ok())
	{
		callback(status);
		return;
	}

	std::vector<int64_t> _tensor_shape;
	std::string _shape2string, _shape2string_2left, _shape2string_2right ;
	for (int i = 0; i < tensor.shape().dims(); i++)
	{
		auto tmp_size = tensor.shape().dim_size(i);
		_tensor_shape.push_back(tensor.shape().dim_size(i));
		_shape2string += ("_" + std::to_string(tmp_size));
	}

	//tensor_table_entry e;
	TensorRingStruct trs;
	trs.left_dtuple = (DataTuple*)malloc(sizeof(DataTuple));
	trs.right_dtuple = (DataTuple*)malloc(sizeof(DataTuple));


	//e.tensor_name = _shape2string + "_" + name;

	trs.tensor_name =   _shape2string + "_" + name;
	_shape2string_2left = trs.tensor_name + "_to_0";
	_shape2string_2right = trs.tensor_name + "_to_1";

	strcpy((trs.left_dtuple)->data_name, _shape2string_2left.c_str());
	strcpy((trs.right_dtuple)->data_name, _shape2string_2right.c_str());


	//e.context = context;


	//if omitted gjk
	//e.tensor = tensor;
	//e.output = output;
#if _show_res__
	printf("allreduce tensor_name is %s ||  %S\n", _shape2string_2left.c_str(), _shape2string_2right.c_str());
#endif
	//e.ready_event = ready_event;
	//e.device = device;
	//e.callback = callback;

	trs.ready_event = ready_event;
	trs.device = device;
	trs.callback = callback;
	trs.output = output;
	trs.left_finished = false;
	trs.right_finished = false;

	(trs.left_dtuple)->data_type = dtype;
	(trs.left_dtuple)->toRight = false;
	(trs.left_dtuple)->op = RING_ALLREDUCE;


	(trs.right_dtuple)->data_type = dtype;
	(trs.right_dtuple)->op = RING_ALLREDUCE;
	(trs.right_dtuple)->toRight = true;


	//e.tensor_data = nullptr;
	(trs.left_dtuple)->data = nullptr;
	(trs.right_dtuple)->data = nullptr;
	char* src_ptr = (char*)(tensor.tensor_data().data());

	auto _type_size = TYPE_SIZE[dtype];
	(trs.left_dtuple)->rank = mring->getRingRank();
	(trs.right_dtuple)->rank = mring->getRingRank();
	//e.gather_tensor.resize(e.available_nums);
	(trs.left_dtuple)->ring_num = mring->getRingNum();
	(trs.right_dtuple)->ring_num = mring->getRingNum();

	int NumofEle = tensor.NumElements();
	(trs.left_dtuple)->data_num = NumofEle / 2;
	(trs.right_dtuple)->data_num = NumofEle  - (trs.left_dtuple)->data_num ;
	//printf("IN OPS   na,e=%s  left_num = %d   right_num =%d\n", trs.tensor_name.c_str(), (trs.left_dtuple)->data_num, (trs.right_dtuple)->data_num);

	(trs.left_dtuple)->data = std::malloc(((trs.left_dtuple)->data_num) * _type_size);
	(trs.right_dtuple)->data = std::malloc(((trs.right_dtuple)->data_num) * _type_size);

	int64 left_sz = ((trs.left_dtuple)->data_num) * _type_size;
	int64 right_sz = ((trs.right_dtuple)->data_num) * _type_size;

	//printf("Reduce device_id = %d\n",  device);
#if HAVE_CUDA
	if (trs.device != CPU_DEVICE_ID)/*for gpu*/
	{
		//printf("Check 2\n");
		cudaStream_t& stream = trs.streams[trs.device];
		//cudaStream_t& stream = mring->streams[trs.device];
		if (stream == nullptr)
		{
			printf("Check 4  trs_name = %s device= %d \n", trs.tensor_name.c_str(), trs.device);
			auto res = check_cuda(trs, "create cuda stream-1",
			                      cudaStreamCreate(&stream));
			if (res == false)
			{
				perror("error in create stream of cuda\n");
				return;
			}
		}
		//printf("enque PollForStatus before\n");
		while (trs.ready_event->PollForStatus() ==
		        perftools::gputools::Event::Status::kPending)
		{
			std::this_thread::sleep_for(std::chrono::nanoseconds(100));
		}
		//printf("enque PollForStatus after\n");
		check_cuda(trs, "memcpy asy from device to host",
		           cudaMemcpyAsync((trs.left_dtuple)->data,
		                           (const void*)src_ptr,
		                           left_sz,
		                           cudaMemcpyDeviceToHost,
		                           stream));
		if (false == check_cuda( trs, "cudaStreamSynchronize asy from device to host", cudaStreamSynchronize(stream)))
			return ;
		char* src_ptr_se = src_ptr + left_sz;
		check_cuda(trs, "memcpy asy from device to host",
		           cudaMemcpyAsync((trs.right_dtuple)->data,
		                           (const void*)src_ptr_se,
		                           right_sz,
		                           cudaMemcpyDeviceToHost,
		                           stream));
		if (false == check_cuda( trs, "cudaStreamSynchronize asy from device to host", cudaStreamSynchronize(stream)))
			return ;
		//printf("Check 6\n");
	}
	else
#endif
	{

		std::memcpy((trs.left_dtuple)->data, src_ptr, left_sz  );
		std::memcpy((trs.right_dtuple)->data, src_ptr + left_sz, (trs.right_dtuple)->data_num * _type_size  );
	}

	{
		/*
		std::lock_guard<std::mutex> enque_lock(bcube_gs.tensor_gene_mutex);
		auto& tensor_table = bcube_gs.tensor_table;
		tensor_table.push(std::move(e));
		**/
		mring->InsertTrs(trs);
	}
	return;
}
// ring must be initialized and the background thread must be running before this function is called.
void ring_allgather_queue(OpKernelContext* context, const Tensor& tensor,
                          GPU_EVENT_IF_CUDA ready_event, const std::string name, const int device,
                          StatusCallback callback)
{

	return;
}
// ring must be initialized and the background thread must be running before this function is called.
void ring_broadcast_queue(OpKernelContext* context, const Tensor& tensor,
                          Tensor* output, int root_rank, GPU_EVENT_IF_CUDA ready_event,
                          const std::string name, const int device, StatusCallback callback)
{

	RING_TYPE dtype;
	Status status = DataTypeToRingType(tensor.dtype(), &dtype);
	if (!status.ok())
	{
		callback(status);
		return;
	}

	std::vector<int64_t> _tensor_shape;
	std::string _shape2string, _shape2string_2left, _shape2string_2right;
	for (int i = 0; i < tensor.shape().dims(); i++)
	{
		_tensor_shape.push_back(tensor.shape().dim_size(i));
	}

	TensorRingStruct trs;
	trs.left_dtuple = (DataTuple*)malloc(sizeof(DataTuple));
	trs.right_dtuple = (DataTuple*)malloc(sizeof(DataTuple));
	//tensor_table_entry e;

	//next part is add shape to distinguish those tensor
	for (size_t ii = 1; ii < _tensor_shape.size(); ii++)
		_shape2string += ("_" + std::to_string(_tensor_shape[ii]));


	//e.tensor_name = _shape2string + "_" + name;
	_shape2string += name;

	//strcpy(trs.tensor_name, _shape2string.c_str());
	trs.tensor_name = _shape2string;
	_shape2string_2left = _shape2string + "_to_0";
	_shape2string_2right = _shape2string + "_to_1";

	strcpy((trs.left_dtuple)->data_name, _shape2string_2left.c_str());
	strcpy((trs.right_dtuple)->data_name, _shape2string_2right.c_str());

	//e.context = context;
	trs.context = context;

	//If omitted gjk
	//e.tensor = tensor;
	/*
		e.ready_event = ready_event;
		e.device = device;
		e.callback = callback;
		e.output = output;
		e.data_type = dtype;
		e.op = RING_BROADCAST;
	**/

	trs.ready_event = ready_event;
	trs.device = device;
	trs.callback = callback;
	trs.output = output;
	trs.left_finished = false;
	trs.right_finished = false;

	//printf("here in ring_broadcast_queue name = %s  output  =%p\n", trs.tensor_name.c_str(), trs.output );
	(trs.left_dtuple)->data_type = dtype;
	(trs.left_dtuple)->toRight = false;
	(trs.left_dtuple)->op = RING_BROADCAST;


	(trs.right_dtuple)->data_type = dtype;
	(trs.right_dtuple)->op = RING_BROADCAST;
	(trs.right_dtuple)->toRight = true;


	{
		/*
		e.block_size = 1;
		e.available_nums = 18;
		e.tensor_size = 18;
		????
		**/
		int element_nums = tensor.NumElements();

		//e.tensor_data = nullptr;
		(trs.left_dtuple)->data = nullptr;
		(trs.right_dtuple)->data = nullptr;

		auto _type_size = TYPE_SIZE[dtype];
		(trs.left_dtuple)->rank = mring->getRingRank();
		(trs.right_dtuple)->rank = mring->getRingRank();
		//e.gather_tensor.resize(e.available_nums);
		(trs.left_dtuple)->ring_num = mring->getRingNum();
		(trs.right_dtuple)->ring_num = mring->getRingNum();

		(trs.left_dtuple)->broadcast_rank = root_rank;
		(trs.right_dtuple)->broadcast_rank = root_rank;
		static std::atomic_int iiiii(1);
		//printf("%2d broadcast tensor_name is %-70s,\telement_nums=%10d, dtype= %d type_size=%d\n", iiiii++, e.e.data_name, element_nums, dtype, _type_size);
		//printf("%2d broadcast tensor_name is %-70s,\telement_nums=%10d, dtype= %d type_size=%d\n", iiiii++, trs.tensor_name.c_str(), element_nums, dtype, _type_size);

		{
			/*
			for (auto& it : e.gather_tensor)
			{
				it.tensor_shape = element_nums;
				it.tensor_ptr = (void*)std::malloc(it.tensor_shape * _type_size);
				assert(it.tensor_ptr != nullptr);
				std::memcpy(it.tensor_ptr, (const void*)tensor.tensor_data().data(),
				            it.tensor_shape * _type_size);
			}
			**/
			(trs.left_dtuple)->data_num = element_nums / 2;
			(trs.right_dtuple)->data_num = element_nums - (trs.left_dtuple)->data_num ;
			(trs.left_dtuple)->data = (void*)std::malloc((trs.left_dtuple)->data_num * _type_size);
			assert((trs.left_dtuple)->data != nullptr);
			//printf("left_data = %p\n", (trs.left_dtuple)->data );
			(trs.right_dtuple)->data = (void*)std::malloc((trs.right_dtuple)->data_num * _type_size);
			assert((trs.right_dtuple)->data != nullptr);


			int64 left_sz = ((trs.left_dtuple)->data_num) * _type_size;
			int64 right_sz = ((trs.right_dtuple)->data_num) * _type_size;
			char* src_ptr = (char*)(tensor.tensor_data().data());
			char* src_ptr_se = src_ptr + left_sz;
			//printf("Determing Whether CUDA?\n");

#if HAVE_CUDA
			//printf("HAVE CUDA  device = %d \n", device);
			//getchar();
			if (trs.device != CPU_DEVICE_ID)/*for gpu*/
			{
				//printf("In Cuda\n");
				cudaStream_t& stream = trs.streams[trs.device];
				//printf("Is here\n");
				//cudaStream_t& stream = mring->streams[trs.device];
				//printf("ok\n");
				if (stream == nullptr)
				{
					printf("Check 4  device = %d\n", trs.device);
					auto res = check_cuda(trs, "create cuda stream",
					                      cudaStreamCreate(&stream));
					if (res == false)
					{
						perror("error in create stream of cuda\n");
						return;
					}
				}
				//printf("enque PollForStatus before\n");
				while (trs.ready_event->PollForStatus() ==
				        perftools::gputools::Event::Status::kPending)
				{
					std::this_thread::sleep_for(std::chrono::nanoseconds(100));
				}
				printf("element_nums= %d  _type_size=%d  src_ptr = %p  left_sz = %ld src_ptr_se = %p  right_sz=%ld\n", element_nums, _type_size, src_ptr, left_sz, src_ptr_se, right_sz);
				check_cuda(trs, "Left memcpy asy from device to host",
				           cudaMemcpyAsync((trs.left_dtuple)->data,
				                           (const void*)src_ptr,
				                           left_sz,
				                           cudaMemcpyDeviceToHost,
				                           stream));
				if (false == check_cuda( trs, "Left cudaStreamSynchronize asy from device to host", cudaStreamSynchronize(stream)))
					return ;

				check_cuda(trs, "Right memcpy asy from device to host",
				           cudaMemcpyAsync((trs.right_dtuple)->data,
				                           (const void*)src_ptr_se,
				                           right_sz,
				                           cudaMemcpyDeviceToHost,
				                           stream));
				if (false == check_cuda( trs, "Right cudaStreamSynchronize asy from device to host", cudaStreamSynchronize(stream)))
					return ;
			}
			else
#endif
			{
				//printf("No CUDA\n");
				//getchar();
				std::memcpy((trs.left_dtuple)->data, src_ptr, left_sz  );
				std::memcpy((trs.right_dtuple)->data, src_ptr + left_sz, (trs.right_dtuple)->data_num * _type_size   );
			}
			//printf("right_data = %p\n", (trs.right_dtuple)->data );

			//e.data = (void*)std::malloc(tensor.tensor_data().size());
			//assert(e.data != nullptr);
		}

	}
	//e.tensor_type = dtype;
	//e.tensor_ops = BROADCAST;

	{
		//std::lock_guard<std::mutex> enque_lock(bcube_gs.tensor_gene_mutex);
		//printf("Before InseertTtrs  name=%s  output=%p  left_data=%p  right_data =%p\n", trs.tensor_name.c_str(), trs.output, trs.left_dtuple->data, trs.right_dtuple->data);
		mring->InsertTrs(trs);
		//mring->EnqueNewQueue2Left(trs.left_dtuple);
		//mring->EnqueNewQueue2Right(trs.right_dtuple);

		//auto& tensor_table = bcube_gs.tensor_table;
		//tensor_table.push(std::move(e));
	}

	return;
}
// C interface to initialize Ring.
extern "C" void ring_tensorflow_init()
{
	/*
	auto &bgs = bcube_gs;
	bcube_all_init_onice(bgs);
	**/
	printf("Init Start\n");
	int ring_num = atoi(getenv("RING_NUM"));
	int ring_rank = atoi(getenv("RING_RANK"));
	printf("%d %d\n", ring_num, ring_rank);
	mring = new MyRing(ring_num, ring_rank);

	while ((!mring->get2LeftConnected()) || (!mring->get2RightConnected()) )
	{
		sleep(1);
	}
	printf("Init End\n");


}

// C interface to get index of current Ring process.
// Returns -1 if Ring is not initialized.
extern "C" int ring_tensorflow_rank()
{
	/*
	if (!bcube_gs.is_inited_done)return -1;
	return bcube_gs.bcube_s.rank;
	**/
	if (NULL == mring)
	{
		return -1;
	}

	return mring->getRingRank();
}

// C interface to get index of current Ring process in the node it is on..
// Returns -1 if Ring is not initialized.
extern "C" int ring_tensorflow_local_rank()
{
	return 0;
}

// C interface to return number of Ring processes.
// Returns -1 if Ring is not initialized.
extern "C" int ring_tensorflow_size()
{
	/*
	if (!bcube_gs.is_inited_done)return -1;
	return bcube_gs.bcube_s.bcube_node_count;
	**/
	if (NULL == mring)
	{
		return -1;
	}

	return mring->getRingNum();
}
int GetDeviceID(OpKernelContext* context)
{
	int device = CPU_DEVICE_ID;
	if (context->device() != nullptr &&
	        context->device()->tensorflow_gpu_device_info() != nullptr)
	{
		device = context->device()->tensorflow_gpu_device_info()->gpu_id;
	}
	return device;
}

// On GPU this event will signal that data is ready, and tensors are
// allocated.
GPU_EVENT_IF_CUDA RecordReadyEvent(OpKernelContext* context)
{
#if HAVE_CUDA
	auto device_context = context->op_device_context();
	if (device_context != nullptr)
	{
		auto executor = device_context->stream()->parent();
		GPU_EVENT_IF_CUDA ready_event = new perftools::gputools::Event(executor);
		ready_event->Init();
		device_context->stream()->ThenRecordEvent(ready_event);
		return ready_event;
	}
#endif
	return nullptr;
}
}//namespace tensorflow

class RingAllreduceOp : public AsyncOpKernel
{
public:
	explicit RingAllreduceOp(OpKernelConstruction* context)
		: AsyncOpKernel(context) {}

	void ComputeAsync(OpKernelContext* context, DoneCallback done) override
	{
		OP_REQUIRES_OK(context, CheckInitialized());

		auto node_name = name();
		auto device = GetDeviceID(context);
		auto tensor = context->input(0);
		Tensor* output;
		OP_REQUIRES_OK(context,
		               context->allocate_output(0, tensor.shape(), &output));
		GPU_EVENT_IF_CUDA ready_event = RecordReadyEvent(context);


		ring_allreduce_queue(context, tensor, output, ready_event, node_name,
		                     device, [context, done](const Status & status)
		{
			context->SetStatus(status);
			done();
		});
	}
};

REGISTER_KERNEL_BUILDER(Name("RingAllreduce").Device(DEVICE_CPU),
                        RingAllreduceOp);
#if T_RING_GPU_ALLREDUCE
REGISTER_KERNEL_BUILDER(Name("RingAllreduce").Device(DEVICE_GPU),
                        RingAllreduceOp);
#endif

REGISTER_OP("RingAllreduce")
.Attr("T: {int32, int64, float32, float64}")
.Input("tensor: T")
.Output("sum: T")
.SetShapeFn([](shape_inference::InferenceContext* c)
{
	c->set_output(0, c->input(0));
	return Status::OK();
})
.Doc(R"doc(
					Perform an Ring Allreduce on a tensor. All other nodes that do a reduction
					on a tensor with the same name must have the same dimension for that tensor.
					Tensors are reduced with other tensors that have the same node name for the
					allreduce.

					Arguments
						tensor:     A tensor to reduce.

					Output
						sum:    A tensor with the same shape as `tensor`, summed across all Ring nodes.
					)doc");





class RingAllgatherOp : public AsyncOpKernel
{
public:
	explicit RingAllgatherOp(OpKernelConstruction* context) : AsyncOpKernel(context) {}

	void ComputeAsync(OpKernelContext* context, DoneCallback done) override
	{
		OP_REQUIRES_OK(context, CheckInitialized());

		auto node_name = name();
		auto device = GetDeviceID(context);
		auto tensor = context->input(0);
		// We cannot pre-allocate output for allgather, since shape of result
		// is only known after all nodes make a notice.
		GPU_EVENT_IF_CUDA ready_event = RecordReadyEvent(context);
		ring_allgather_queue(context, tensor, ready_event, node_name, device,
		                     [context, done](const Status & status)
		{
			context->SetStatus(status);
			done();
		});
	}
};

REGISTER_KERNEL_BUILDER(Name("RingAllgather").Device(DEVICE_CPU),
                        RingAllgatherOp);
#if RING_GPU_ALLGATHER
REGISTER_KERNEL_BUILDER(Name("RingAllgather").Device(DEVICE_GPU),
                        RingAllgatherOp);
#endif

REGISTER_OP("RingAllgather")
.Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64, bool}")
.Input("tensor: T")
.Output("output: T")
.SetShapeFn([](shape_inference::InferenceContext* c)
{
	shape_inference::ShapeHandle output;
	TF_RETURN_IF_ERROR(
	    c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
	c->set_output(0, output);
	return Status::OK();
})
.Doc(R"doc(
					Perform an Ring Allgather on a tensor. All other processes that do a gather on a
					tensor with the same name must have the same rank for that tensor, and have the
					same dimension on all but the first dimension.
					Arguments
						tensor:     A tensor to gather.
					Output
						gathered:    A tensor with the same shape as `tensor` except for the first dimension.
					)doc");


class RingBroadcastOp : public AsyncOpKernel
{
public:
	explicit RingBroadcastOp(OpKernelConstruction* context) : AsyncOpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank_));
	}

	void ComputeAsync(OpKernelContext* context, DoneCallback done) override
	{
		OP_REQUIRES_OK(context, CheckInitialized());

		auto node_name = name();
		auto device = GetDeviceID(context);
		auto tensor = context->input(0);
		Tensor* output = nullptr;

		OP_REQUIRES_OK(context, context->allocate_output(0, tensor.shape(), &output));
		GPU_EVENT_IF_CUDA ready_event = RecordReadyEvent(context);
		ring_broadcast_queue(context, tensor, output, root_rank_, ready_event, node_name, device,
		                     [context, done](const Status & status)
		{
			//printf("Before Set Context\n");
			context->SetStatus(status);
			//printf("Doing\n");
			done();
			//printf("After Done\n");
		});
	}

private:
	int root_rank_;
};

REGISTER_KERNEL_BUILDER(Name("RingBroadcast").Device(DEVICE_CPU),
                        RingBroadcastOp);
#if T_RING_GPU_BROADCAST
REGISTER_KERNEL_BUILDER(Name("RingBroadcast").Device(DEVICE_GPU),
                        RingBroadcastOp);
#endif

REGISTER_OP("RingBroadcast")
.Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64, bool}")
.Attr("root_rank: int")
.Input("tensor: T")
.Output("output: T")
.SetShapeFn([](shape_inference::InferenceContext* c)
{
	c->set_output(0, c->input(0));
	return Status::OK();
})
.Doc(R"doc(
					Perform an Ring Broadcast on a tensor. All other processes that do a broadcast
					on a tensor with the same name must have the same dimension for that tensor.
					Arguments
						tensor:     A tensor to broadcast.
						root_rank:  Rank that will send data, other ranks will receive data.
					Output
						output:    A tensor with the same shape as `tensor` and same value as
								   `tensor` on root rank.
					)doc");









}//namespace tensorflow
}//namespce bcube
