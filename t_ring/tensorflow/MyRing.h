#ifndef MY_RING_H
#define MY_RING_H

#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <queue>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#define TENSORFLOW 1

#define HAVE_NCCL 0
#define QueueLen 1000

//#if TENSORFLOW
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "rdma_t.h"

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

//#endif



#if HAVE_CUDA
#include "tensorflow/stream_executor/stream.h"
#include <cuda_runtime.h>
#endif

#if HAVE_NCCL
#include <nccl.h>
#endif
#define DATA_NAME_LEN 300

#if HAVE_CUDA
#define GPU_EVENT_IF_CUDA perftools::gputools::Event*
#else
#define GPU_EVENT_IF_CUDA void*
#endif

enum RING_OP {RING_BROADCAST = 1, RING_ALLGATHER, RING_ALLREDUCE };
typedef enum
{
	T_VOID    = 0,  T_BOOL    = 1,
	T_UINIT8  = 2,  T_INIT8   = 3,
	T_UINT16  = 4,  T_INT16   = 5,
	T_UINT32  = 6,  T_INT32   = 7,
	T_UINT64  = 8,  T_INT64   = 9,
	T_FLOAT32 = 10, T_FLOAT64 = 11
} TENSOR_TYPE;

typedef TENSOR_TYPE RING_TYPE;
#define MAX_RING_NUM 20
struct DataTuple
{
	int rank;
	int broadcast_rank;
	char data_name[DATA_NAME_LEN];
	int scatter_gather_counter;
	int start_idx;
	bool toRight;
	RING_OP op;
	RING_TYPE data_type;
	int data_num;
	void* data;
	void* replica_ptrs[MAX_RING_NUM];
	int ring_num;

};

struct TensorRingStruct
{
	string tensor_name;
	DataTuple* left_dtuple;
	DataTuple* right_dtuple;
	bool left_finished;
	bool right_finished;
#ifdef __have__tensorflow__
	OpKernelContext* context;/*context for tensor*/
	Tensor tensor;/*Input tensor.*/
	Tensor* output;/* Pre-allocated output tensor. */
	int device;
	GPU_EVENT_IF_CUDA ready_event;// Event indicating that data is ready.
	Status_callback callback;
#endif
};

static  int TYPE_SIZE[] =
{
	4,				sizeof(bool),
	sizeof(uint8_t), sizeof(int8_t),
	sizeof(uint16_t), sizeof(int16_t),
	sizeof(uint32_t), sizeof(int32_t),
	sizeof(uint64_t), sizeof(int64_t),
	sizeof(float_t), sizeof(double_t)
};



#define IP_ADDR "127.0.0.1"
#define BASE_PORT 7000
#define MAX_NUM 9
using namespace std;

class MyRing
{
public:
	MyRing(int rn, int rr);

	int sizeoftype(RING_TYPE dt);
	void OutPutTrs();
	int getRingNum();
	int getRingRank();
	bool get2LeftConnected();
	bool get2RightConnected();
	int getLeftNeighbour(int my_rank);
	int getRightNeighbour(int my_rank);
	void EnqueNewQueue2Left(DataTuple* dtuple);
	void EnqueNewQueue2Right(DataTuple* dtuple);
	void InsertTrs(TensorRingStruct& trs);
	void InitBGThread();
	int InitConnection(char* local_ip, char* remote_ip, int remote_port); //as client

	void Send2RightThreadCallback();
	void Send2LeftThreadCallback();
	void Recv4LeftThreadCallback();
	void Recv4RightThreadCallback();
	int Wait4Connection(int bind_port); // as server


	char* RecvFixedData(int connected_fd, size_t len);
	void ProcessRecvData(int connected_fd);


	void BackGround2LeftThreadCallback();
	void BackGround2RightThreadCallback();
	bool isScatterStage(int stage_id);
	void EnqueSendQ(DataTuple* dt);
	void OutPutTuple(void* dataTuple, bool freeMem);
	void FinishedTuple(void* dataTuple,  bool freeMem);
	void ProcessStageData(void* local_data, void* recv_data, int cur_statge);
	void MergeData(void* local_buf, void* recvbuf, bool isScatter);
	void RingAllReduceMerge(DataTuple* recvTuple, DataTuple* localTuple, bool isScatter);
	void RingAllGatherMerge(DataTuple* recvTuple, DataTuple* localTuple);
	void RingBroadCastMerge(DataTuple* recvTuple, DataTuple* localTuple);
	void Release_src(TensorRingStruct* trs, bool freeMem);
	void FreeDataTuple(DataTuple*& dtuple);
	void* GenAllReduceBuf(DataTuple* dtuple);
	void* GenAllGatherBuf(DataTuple* dtuple);
	void* GenBroadCastBuf(DataTuple* dtuple);
	//string GenUniqueKey(DataTuple: dtuple);
	string getOp(RING_OP op);
	static void EnqueQueue(size_t thread_rank, int ele_num, bool toRight, RING_OP r_op);
	void bench_test(size_t thread_num);
	void ShutDown();

#if HAVE_RDMA
	struct rdma_cm_id* RDMA_InitConnection(char* local_ip, char* remote_ip, int remote_port); //as client
	struct rdma_cm_id* RDMA_Wait4Connection(int bind_port); //as server
	void RDMA_ProcessRecvData(struct rdma_cm_id* rc_id);
	void RDMA_RecvFixedData(struct rdma_cm_id* rc_id, size_t len);
	void* FetchFrom2RightQ();
	void* FetchFrom2LeftQ();
#endif

	~MyRing();
private:
	static int ring_rank;
	static int ring_num;
	static int stage_num;

	static int scatter_gather_counter;
	//static int from_left_queue_front;
	//static int from_left_queue_tail;
	static int to_right_queue_front;
	static int to_right_queue_tail;
	//static int from_right_queue_front;
	//static int from_right_queue_tail;
	static int to_left_queue_front;
	static int to_left_queue_tail;

	static bool to_right_connected;
	static bool to_left_connected;

	static bool shut_down;
	//static std::mutex mtx;
	//static map<string, void*> recv_buf_map;
	static const int header_name_len = DATA_NAME_LEN;
	constexpr static  char* ip_arrs[MAX_NUM] = { (char*)"192.168.13.246", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1"};


	constexpr static  char* to_right_ip_arrs[MAX_NUM] = { (char*)"192.168.13.246", (char*)"192.168.13.183", (char*)"192.168.13.27", (char*)"192.168.13.25", (char*)"192.168.13.23", (char*)"192.168.13.176", (char*)"192.168.13.179", (char*)"192.168.13.24", (char*)"192.168.13.26"};
	constexpr static  char* to_left_ip_arrs[MAX_NUM] = { (char*)"192.168.13.44", (char*)"192.168.13.41", (char*)"192.168.13.43", (char*)"192.168.13.47", (char*)"192.168.13.45", (char*)"192.168.13.40", (char*)"192.168.13.42", (char*)"192.168.13.46", (char*)"192.168.13.39"};

	//constexpr static  char* to_right_ip_arrs[MAX_NUM] = { (char*)"12.12.10.11", (char*)"12.12.10.12", (char*)"12.12.10.13", (char*)"12.12.10.14", (char*)"12.12.10.15", (char*)"12.12.10.16", (char*)"12.12.10.17", (char*)"12.12.10.18", (char*)"12.12.10.19"};
	//constexpr static  char* to_left_ip_arrs[MAX_NUM] = { (char*)"12.12.11.11", (char*)"12.12.11.12", (char*)"12.12.11.13", (char*)"12.12.11.14", (char*)"12.12.11.15", (char*)"12.12.11.16", (char*)"12.12.11.17", (char*)"12.12.11.18", (char*)"12.12.11.19"};

	constexpr static  char* rdma_to_right_ip_arrs[MAX_NUM] = { (char*)"12.12.10.11", (char*)"12.12.10.12", (char*)"12.12.10.13", (char*)"12.12.10.14", (char*)"12.12.10.15", (char*)"12.12.10.16", (char*)"12.12.10.17", (char*)"12.12.10.18", (char*)"12.12.10.19"};
	constexpr static  char* rdma_to_left_ip_arrs[MAX_NUM] = { (char*)"12.12.11.11", (char*)"12.12.11.12", (char*)"12.12.11.13", (char*)"12.12.11.14", (char*)"12.12.11.15", (char*)"12.12.11.16", (char*)"12.12.11.17", (char*)"12.12.11.18", (char*)"12.12.11.19"};

	const static int listen_for_left_connection_port = 9111; // to right ip
	const static int listen_for_right_connection_port = 9112;

	const static int rdma_listen_for_left_connection_port = 12345; // to right ip
	const static int rdma_listen_for_right_connection_port = 12346;

	static std::queue<void*> to_right_queue;
	static std::mutex right_queue_mtx;
	static std::queue<void*> to_left_queue;
	static std::mutex left_queue_mtx;
	static std::mutex out_mutex;

	//static std::queue<void*> new_queue;
	//static std::mutex new_queue_mtx;



	static std::mutex map_mtx_to_left;
	static map<string, void*> recv_buf_map_to_left;

	static std::mutex map_mtx_to_right;
	static map<string, void*> recv_buf_map_to_right;

	static std::queue<void*>new_queue_to_left;
	static std::mutex new_queue_to_left_mutex;

	static std::queue<void*>new_queue_to_right;
	static std::mutex new_queue_to_right_mutex;

	//static vector<queue<void*>> process_queues;

	static vector<queue<void*>> process_queues_to_left;
	static vector<queue<void*>> process_queues_to_right;


	static std::mutex trs_mutex;
	static map<string, TensorRingStruct> trs_map;

	static std::vector<std::thread> thread_vec;

	//static void* from_left_queue[QueueLen];
	//static void* to_right_queue[QueueLen];
	//static void* from_right_queue[QueueLen];
	//static void* to_left_queue[QueueLen];

	/*
		queue<void*> from_left_queue;
		queue<void*> to_right_queue;

		//reserve for further extension
		queue<void*> from_right_queue;
		queue<void*> to_left_queue;
	**/


};

#endif