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

#include <mutex>
#include <stdexcept>
#include <sstream>
#define TENSORFLOW 0
#define HAVE_CUDA 0
#define HAVE_NCCL 0
#define QueueLen 1000
#if TENSORFLOW
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#endif

#define EIGEN_USE_THREADS

#if HAVE_CUDA
#include "tensorflow/stream_executor/stream.h"
#include <cuda_runtime.h>
#endif

#if HAVE_NCCL
#include <nccl.h>
#endif
#define DATA_NAME_LEN 100
enum DataType { FLOAT32 = 1, FLOAT64, INTEGER, LONGINT};
enum RING_OP {RING_BROADCAST = 1, RING_ALLGATHER, RING_ALLREDUCE };
struct  Tensor
{
	void* data;
};
struct DataTuple
{
	int rank;
	int broadcast_rank;
	char data_name[DATA_NAME_LEN];
	int scatter_gather_counter;
	int start_idx;
	bool toRight;
	RING_OP op;
	DataType data_type;
	int data_num;
	void* data;
	std::vector<void*> replica_ptrs;

};
#define IP_ADDR "127.0.0.1"
#define BASE_PORT 7000
#define MAX_NUM 9
using namespace std;

class MyRing
{
public:

	MyRing(int rn, int rr);

	int sizeoftype(DataType dt);

	int getLeftNeighbour(int my_rank);
	int getRightNeighbour(int my_rank);
	void InitBGThread();
	int InitConnection(char* ip_addr, int port);
	void Send2RightThreadCallback();
	void Send2LeftThreadCallback();
	void Recv4LeftThreadCallback();
	void Recv4RightThreadCallback();
	int Wait4Connection(int bind_port);
	char* RecvFixedData(int connected_fd, size_t len);
	void ProcessRecvData(int connected_fd);
	//void BackGroundThreadCallback();
	void BackGround2LeftThreadCallback();
	void BackGround2RightThreadCallback();
	bool isScatterStage(int stage_id);
	void EnqueSendQ(DataTuple* dt);
	void OutPutTuple(void* dataTuple);
	void ProcessStageData(void* local_data, void* recv_data, int cur_statge);
	void MergeData(void* local_buf, void* recvbuf, bool isScatter);
	void RingAllReduceMerge(DataTuple* recvTuple, DataTuple* localTuple, bool isScatter);
	void RingAllGatherMerge(DataTuple* recvTuple, DataTuple* localTuple);
	void RingBroadCastMerge(DataTuple* recvTuple, DataTuple* localTuple);
	void FreeDataTuple(DataTuple* dtuple);
	void* GenAllReduceBuf(DataTuple* dtuple);
	void* GenAllGatherBuf(DataTuple* dtuple);
	void* GenBroadCastBuf(DataTuple* dtuple);
	void EnqueReduceQueue(int ele_num);
	void Reduce_test(int thread_num);
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


	//static std::mutex mtx;
	//static map<string, void*> recv_buf_map;
	static const int header_name_len = 100;
	constexpr static  char* ip_arrs[MAX_NUM] = { (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1", (char*)"127.0.0.1"};
	constexpr static  int from_left_port_arrs[MAX_NUM] = {7000, 7002, 7004, 7006, 7008, 7010, 7012, 7014, 7016};
	constexpr static  int from_right_port_arrs[MAX_NUM] = {7001, 7003, 7005, 7007, 7009, 7011, 7013, 7015, 7017};

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