#include "MyRing.h"
//map<string, void*> MyRing::recv_buf_map;
//std::mutex MyRing:: mtx;
const int MyRing::header_name_len;
constexpr char* MyRing::ip_arrs[MAX_NUM];


constexpr char* MyRing::to_right_ip_arrs[MAX_NUM];
constexpr char* MyRing::to_left_ip_arrs[MAX_NUM];
const int MyRing::listen_for_left_connection_port;
const int MyRing::listen_for_right_connection_port;

//#define GJK_DEBUG 1
int MyRing::ring_rank;
int MyRing::ring_num;
int MyRing::scatter_gather_counter;
int MyRing::to_right_queue_front;
int MyRing::to_right_queue_tail;
int MyRing::to_left_queue_front;
int MyRing::to_left_queue_tail;
int MyRing::stage_num;
std::queue<void*> MyRing::to_right_queue;
std::mutex MyRing::right_queue_mtx;
std::queue<void*> MyRing::to_left_queue;
std::mutex MyRing::left_queue_mtx;

std::mutex MyRing::out_mutex;

std::mutex MyRing::map_mtx_to_left;
map<string, void*> MyRing::recv_buf_map_to_left;

std::mutex MyRing::map_mtx_to_right;
map<string, void*> MyRing::recv_buf_map_to_right;

std::queue<void*> MyRing::new_queue_to_left;
std::mutex MyRing::new_queue_to_left_mutex;

std::queue<void*> MyRing::new_queue_to_right;
std::mutex MyRing::new_queue_to_right_mutex;

//vector<queue<void*>> MyRing::process_queues;

vector<queue<void*>> MyRing::process_queues_to_left;
vector<queue<void*>> MyRing::process_queues_to_right;

std::vector<std::thread> MyRing::thread_vec;


std::mutex MyRing::trs_mutex;
map<string, TensorRingStruct> MyRing::trs_map;

//void MyRing::EnqueReduceQueue(int ele_num, bool toRight);
//std::queue<void*> MyRing::new_queue;
//std::mutex MyRing::new_queue_mtx;
//void* MyRing::to_right_queue[QueueLen];
//void* MyRing::to_left_queue[QueueLen];
bool MyRing::to_right_connected;
bool MyRing::to_left_connected;
bool MyRing::shut_down;

MyRing::MyRing(int rn, int rr)
{
	//just for test
	this->ring_num = rn;
	this->ring_rank = rr % this->ring_num;

	this->stage_num = (this->ring_num - 1) * 2;

	to_right_connected = false;
	to_left_connected = false;
	shut_down = false;
	//process_queues.resize(this->stage_num);
	process_queues_to_right.resize(this->stage_num);
	process_queues_to_left.resize(this->stage_num);

	printf("ring_num = %d rank = %d  \n", this->ring_num, this->ring_rank );
	printf("to_left_ip=%s to_right_ip=%s  \n", this->to_left_ip_arrs[ring_rank], this->to_right_ip_arrs[ring_rank]);
	printf("listen_for_right_connection_port=%d listen_for_left_connection_port=%d", this->listen_for_right_connection_port, this->listen_for_left_connection_port);
	this->InitBGThread();
	printf("Finished InitBG\n");
}
void MyRing::OutPutTrs()
{
	printf("++++++++++++++++++++++++++++++++\n");
	printf("MAP_SZ %d\n", trs_map.size() );
	//std::lock_guard<std::mutex> lock(trs_mutex);
	//map<string, void*> trs_map1 = trs_map;
	std::map<string,  TensorRingStruct>::iterator iter = trs_map.begin();
	/*
	while (vit != trs_map.end())
	{
		TensorRingStruct* titem = static_cast<TensorRingStruct*>(vit->second);
		printf("na = %s   out = %p\n", titem->tensor_name.c_str(), titem->output);
		vit++;

	}**/
	for (iter = trs_map.begin(); iter != trs_map.end(); iter++)
	{
		//std::cout << iter->first << " : " << (iter->second).output << std::endl;

		DataTuple* ll = (iter->second).left_dtuple;
		DataTuple* rr = (iter->second).right_dtuple;
		printf("%s   %p  %p  %p\n", (iter->second).tensor_name.c_str(), (iter->second).output,  ll->data, rr->data );
	}

	printf("++++++++++++++++++++++++++++++++\n");
}
bool MyRing::get2LeftConnected()
{
	return this->to_left_connected;
}
bool MyRing::get2RightConnected()
{
	return this->to_right_connected;
}
int MyRing::getRingNum()
{
	return this->ring_num;
}
int MyRing::getRingRank()
{
	return this->ring_rank;
}
int MyRing::getLeftNeighbour(int myrank)
{
	int rank = myrank - 1;
	while (rank < 0)
	{
		rank += ring_num;
	}
	return (rank) % (ring_num);
}
int MyRing::getRightNeighbour(int myrank)
{
	int rank = myrank + 1;
	while (rank < 0)
	{
		rank += ring_num;
	}
	return (rank) % (ring_num);
}
/*
DataTuple* MyRing::EncodeData(){

}
**/
void MyRing::InsertTrs(TensorRingStruct& trs)
{
#ifdef GJK_DEBUG
	printf("Enter Check=\n");
#endif
	{
		std::lock_guard<std::mutex> lock(trs_mutex);

		//printf("Before Insert name = %s    output=%p  left_data=%p  right_data =%p\n", trs.tensor_name.c_str(), trs.output, trs.left_dtuple->data, trs.right_dtuple->data);
		//OutPutTrs();
		string kstr = trs.tensor_name;
		std::map<string, TensorRingStruct>::iterator vit = trs_map.find(kstr);
		//assert(vit == trs_map.end());
		if (vit == trs_map.end())
		{
			trs_map.insert(make_pair(kstr, std::move(trs)));
		}

		//printf("Insert name= %s  output2= %p\n", trs.tensor_name.c_str(), trs.output );
		//getchar();

		//OutPutTrs();
	}

	EnqueNewQueue2Left(trs.left_dtuple);
	EnqueNewQueue2Right(trs.right_dtuple);
}

void MyRing::FinishedTuple(void* dtp,  bool freeMem = true)
{
#ifdef GJK_DEBUG
	printf("Finished Tuple\n");
#endif
	DataTuple* dataTuple = static_cast<DataTuple*>(dtp);
	char tensor_name[DATA_NAME_LEN];
	int len = strlen(dataTuple->data_name);
	len -= 5; //_to_0
	strncpy(tensor_name, (dataTuple->data_name), len );
	tensor_name[len] = '\0';
#ifdef GJK_DEBUG
	printf("tensor_name = %s\n", tensor_name);
#endif
	TensorRingStruct* trs_ptr = NULL;
	bool canRet = false;
	string kstr = tensor_name;
	std::map<string, TensorRingStruct>::iterator vit;

	/*

		{
			std::lock_guard<std::mutex> lock(trs_mutex);
			printf("++++++++++++++++++++++++++++++++\n");
			printf("MAP_SZ %d\n", trs_map.size() );
			std::map<string,  TensorRingStruct>::iterator iter = trs_map.begin();

			for (iter = trs_map.begin(); iter != trs_map.end(); iter++)
			{
				//std::cout << iter->first << " : " << (iter->second).output << std::endl;

				DataTuple* ll = (iter->second).left_dtuple;
				DataTuple* rr = (iter->second).right_dtuple;
				printf("%s   %p  %p  %p\n", (iter->second).tensor_name.c_str(), (iter->second).output,  ll->data, rr->data );
			}

			printf("++++++++++++++++++++++++++++++++\n");
		}


	**/

	{
		std::lock_guard<std::mutex> lock(trs_mutex);



#ifdef GJK_DEBUG
		printf("FIN Check 0.1\n");
		std::cout << "thisis \t" << kstr << "\n";

		//getchar();
#endif
		//vit = trs_map.find("test");
		//printf("Test OK\n");

		vit = trs_map.find(kstr);

		//printf("FIN CHECK 0.15\n");
		assert(vit != trs_map.end());

		if (vit != trs_map.end())
		{
#ifdef GJK_DEBUG
			printf("FIN Check-0.2 = \n");
#endif
			//trs = static_cast<TensorRingStruct*>(vit->second);
			auto& trs = vit->second;
			if (dataTuple->toRight)
			{
				trs.right_finished = true;
			}
			else
			{

				trs.left_finished = true;
			}
			if (trs.left_finished  && trs.right_finished)
			{
				canRet = true;
				trs_ptr = &trs;

			}

		}

	}

//if (trs->left_finished && trs->right_finished)
	if (canRet)
	{
#ifdef GJK_DEBUG
		printf("FIN Check-0.9\n");
#endif
		//switch (trs->left_dtuple->op)
		switch (trs_ptr->left_dtuple->op)
		{
		case RING_ALLREDUCE:
		case RING_BROADCAST:
		{
//#ifdef GJK_DEBUG
			//printf("FIN Check-1/0  name=%s  dtp-name=%s output =%p\n", trs_ptr->tensor_name.c_str(), dataTuple->data_name, trs_ptr->output);

//#endif
			char* raw_data = (char*)(trs_ptr->output->tensor_data().data());



#ifdef GJK_DEBUG
			printf("FIN Check-1.9\n");
#endif

			size_t left_sz = (trs_ptr->left_dtuple->data_num) * sizeoftype(trs_ptr->left_dtuple->data_type);
			size_t right_sz = (trs_ptr->right_dtuple->data_num) * sizeoftype(trs_ptr->right_dtuple->data_type);
#ifdef GJK_DEBUG
			printf("FIN Check1 =\n");
#endif
			memcpy( raw_data, trs_ptr->left_dtuple->data, left_sz );
#ifdef GJK_DEBUG
			printf("FIN Check2 =\n");
#endif
			memcpy(raw_data + left_sz, trs_ptr->right_dtuple->data, right_sz);
#ifdef GJK_DEBUG
			printf("FIN Check3 =\n");
#endif
			break;
		}

		case RING_ALLGATHER:
		{

			printf("RING_ALLGATHER  to be completed\n");
			/* all_gather need the shape of every tensor, which has not been included in the ring framework
			size_t data_sz = 0;
			int k = 0;
			DataTuple* ldt = trs->left_dtuple;
			DataTuple* rdt = trs->right_dtuple;

			TensorShape tensor_shape, single_slice_shape;
			int64_t dims_without_0 = 0;
			std::vector<int64_t> tensor_sz;

			for (k = 0; k < this->ring_num; k++)
			{
				DataTuple* lditem = static_cast<DataTuple*>(ldt->replica_ptrs[k]);
				DataTuple* rditem = static_cast<DataTuple*>(rdt->replica_ptrs[k]);
				int64_t ldz = lditem->data_num * sizeoftype(lditem->data_type);
				int64_t rdz = rditem->data_num * sizeoftype(rditem->data_type);
				int64_t dz = ldz + rdz;
				tensor_sz.push_back(dz);
			}
			for (auto& res : tensor_sz)
			{
				dims_without_0 += res;
			}
			for (size_t index = 1; index < tensor_sz.size(); index++)
			{
				single_slice_shape.AddDim(tensor_sz[index]);
			}
			tensor_shape.AddDim(dims_without_0);
			tensor_shape.AppendShape(single_slice_shape);

			status = trs->context->allocate_output(0, tensor_shape, &(trs->output));
			if (!status.ok())
			{
				trs->callback(status);
				return;
			}
			#if HAVE_CUDA
			// On GPU allocation is asynchronous, we need to wait for it to complete.
			auto device_context = e.context->op_device_context();
			if (device_context != nullptr)
			{
				device_context->stream()->BlockHostUntilDone();
			}
			#endif
			char* dst_ptr = (char*)(trs->output->tensor_data().data());
			for (auto& _a_ : e.gather_tensor)
			{
				std::memcpy(dst_ptr, _a_.tensor_ptr, _a_.tensor_shape);
				dst_ptr += _a_.tensor_shape;
			}
			**/
			break;
		}
		}
#ifdef GJK_DEBUG
		printf("FIN before releasing \n");
#endif


		{
			std::lock_guard<std::mutex> lock(trs_mutex);
			vit = trs_map.find(trs_ptr->tensor_name);
			if (vit != trs_map.end())
			{
				trs_ptr = &(vit->second);
				trs_ptr->callback(Status::OK());
				Release_src(trs_ptr, freeMem);
				trs_map.erase(vit);
			}

		}
#ifdef GJK_DEBUG
		printf("Finished ... %s\n", tensor_name);
#endif


		//OutPutTuple(dataTuple, freeMem);
	}

}


void MyRing::EnqueNewQueue2Left(DataTuple* dtuple)
{
#ifdef GJK_DEBUG
	printf("Enter Left  %d\n", new_queue_to_left.size() );
	//OutPutTrs();
#endif
	std::lock_guard<std::mutex> lock(new_queue_to_left_mutex);
	new_queue_to_left.push(dtuple);
#ifdef GJK_DEBUG
	printf("Enque Left  %d\n", new_queue_to_left.size() );
#endif
}
void MyRing::EnqueNewQueue2Right(DataTuple* dtuple)
{
#ifdef GJK_DEBUG
	printf("Enter right  %d\n", new_queue_to_right.size() );
	//OutPutTrs();
#endif
	std::lock_guard<std::mutex> lock(new_queue_to_right_mutex);
	new_queue_to_right.push(dtuple);
#ifdef GJK_DEBUG
	printf("Enque right  %d\n", new_queue_to_right.size() );
#endif
}
void MyRing::EnqueQueue(size_t thread_rank, int ele_num, bool toRight, RING_OP r_op)
{
	int cnt = 0;
	std::ostringstream ss;
	ss << std::this_thread::get_id();
	std::string idstr = ss.str();
	while ( !(to_right_connected &&  to_left_connected) )
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	while (1 == 1)
	{
		srand((unsigned)time(NULL));
		DataTuple* dtuple = (DataTuple*)malloc(sizeof(DataTuple));
		//new(dtuple)DataTuple();//调用一下定位new即可。
		//vector容器的析构函数不用自己调用,系统会进行析构,数据超出作用域后会自动析构
		stringstream ss;

		sprintf(dtuple->data_name, "Tensor_%ld_%d", thread_rank , cnt);

		//printf("OutPUT(%s):\n", idstr.c_str());
		//strcpy(dtuple->data_name, );
		dtuple->start_idx = 0;
		dtuple->rank = ring_rank;
		dtuple->broadcast_rank = 0;
		dtuple->toRight = toRight;
		dtuple->data_type = T_FLOAT32;
		dtuple->data_num = ele_num;
		dtuple->op = r_op;
		dtuple->ring_num = ring_num;
		//dtuple->replica_ptrs = (void**)malloc(sizeof(void*)* ring_num);
		//assert(dtuple->replica_ptrs != nullptr);
		int i = 0;
		for (i = 0; i < ring_num; i++ )
		{
			dtuple->replica_ptrs[i] = NULL;
		}
		dtuple->replica_ptrs[dtuple->rank] = static_cast<void*>(dtuple);

		float* fdata = (float*)malloc(sizeof(float) * (dtuple->data_num));

		for (i = 0; i < dtuple->data_num; i++ )
		{
			fdata[i] =  rand() % 100;
		}
		dtuple->data = static_cast<void*>(fdata);
		//new_queue.push(_2right_dtuple);
		if (toRight)
		{
			new_queue_to_right.push(dtuple);
		}
		else
		{
			new_queue_to_left.push(dtuple);

		}
		cnt++;
		if (cnt > 100)
		{
			break;
		}
		printf("Enqueue  %s\n", dtuple->data_name);
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}


}
void MyRing::bench_test(size_t thread_num)
{
	size_t i = 0;
	std::vector<std::thread> vec;
	for ( i = 0; i < thread_num * 3; i += 3)
	{
		srand((unsigned)time(NULL));
		int ele_num = 320000;
		bool toRight = (i % 2 == 1);

		std::thread titem(MyRing::EnqueQueue, i,  ele_num, toRight, RING_BROADCAST);
		vec.push_back(std::move(titem));
		printf("Thread %ld started\n", i );

		std::thread titem2(MyRing::EnqueQueue, i + 1, ele_num, toRight, RING_ALLGATHER);
		vec.push_back(std::move(titem2));
		printf("Thread %ld started\n", i + 1 );

		std::thread titem3(MyRing::EnqueQueue, i + 2, ele_num, toRight, RING_ALLREDUCE);
		vec.push_back(std::move(titem3));
		printf("Thread %ld started\n", i + 2 );

		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	for (i = 0; i < vec.size(); i++)
	{

		vec[i].join();
	}
	printf("FIN ERUQNWE\n");
}
void MyRing::InitBGThread()
{

	/*
		std::thread td3(&MyRing::Recv4LeftThreadCallback, this);
		std::thread td4(&MyRing::Recv4RightThreadCallback, this);
		std::thread td1(&MyRing::Send2RightThreadCallback, this);
		std::thread td2(&MyRing::Send2LeftThreadCallback, this);
		//std::thread td5(&MyRing::BackGroundThreadCallback, this);
		std::thread td5(&MyRing::BackGround2RightThreadCallback, this);
		std::thread td6(&MyRing::BackGround2LeftThreadCallback, this);
	**/
	thread_vec.push_back(std::thread(&MyRing::Recv4LeftThreadCallback, this));
	thread_vec.push_back(std::thread(&MyRing::Recv4RightThreadCallback, this));
	thread_vec.push_back(std::thread(&MyRing::Send2RightThreadCallback, this));
	thread_vec.push_back(std::thread(&MyRing::Send2LeftThreadCallback, this));
	thread_vec.push_back(std::thread(&MyRing::BackGround2RightThreadCallback, this));
	thread_vec.push_back(std::thread(&MyRing::BackGround2LeftThreadCallback, this));

	sleep(1);
	//bench_test(10);
	printf("Enque Success\n");
	/*
		td1.join();
		td2.join();
		td3.join();
		td4.join();
		td5.join();
		td6.join();
	**/

}
int MyRing::InitConnection(char* local_ip, char* remote_ip, int remote_port)
{
	printf("InitConnection  %s %s  %d\n", local_ip, remote_ip, remote_port );
	struct sockaddr_in ser_in, local_in;/*server ip and local ip*/
	memset(&ser_in, 0, sizeof(ser_in));
	memset(&local_in, 0, sizeof(local_in));

	int tmp_skfd = socket(AF_INET, SOCK_STREAM, 0);/*local socket*/

	/*bind remote socket*/
	ser_in.sin_family = AF_INET;
	ser_in.sin_port = htons(remote_port);/*connect to public port remote*/
	inet_pton(AF_INET, remote_ip, &ser_in.sin_addr);

	/*bind local part*/
	local_in.sin_family = AF_INET;
	//local_in.sin_port = htons();/*server listen public ports*/
	//local_in.sin_addr.s_addr = INADDR_ANY;/*listen any connects*/
	inet_pton(AF_INET, local_ip, &local_in.sin_addr);



	while (bind(tmp_skfd, (struct sockaddr*) & (local_in), sizeof(local_in)) != 0)
	{
		printf("error in bind local addr\n");
		std::this_thread::sleep_for(std::chrono::seconds(1));
		//exit(-1);
	}

	if (1)/*connect to server*/
	{
		int connect_count = 0;
		while (connect(tmp_skfd, (struct sockaddr*) & (ser_in), sizeof(ser_in)) != 0)
		{
			printf("COnnected Failed\n");
			//std::cerr << "waiting to connect to server...." << ++connect_count << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));


		}

	}
	return tmp_skfd;
}
bool MyRing::isScatterStage(int stage_id)
{
	if (stage_id < this->ring_num - 1)
	{
		return true;
	}
	else
	{
		return false;
	}
}
string MyRing::getOp(RING_OP op)
{
	switch (op)
	{
	case RING_ALLREDUCE:
		return "RING_ALLREDUCE";
	case RING_BROADCAST:
		return "RING_BROADCAST";
	case RING_ALLGATHER:
		return "RING_ALLGATHER";
	}
	return "";
}

void MyRing::Release_src(TensorRingStruct* trs, bool freeMem = false)
{
	DataTuple* left_dtuple =  trs->left_dtuple;
	DataTuple* right_dtuple = trs->right_dtuple;
#ifdef GJK_DEBUG
	printf("Release_src check1=\n");
#endif
	if (left_dtuple->op == RING_ALLGATHER)
	{
		int i = 0;
		for (i = 0; i < left_dtuple->ring_num; i++)
		{
			DataTuple* ditem = static_cast<DataTuple*>(left_dtuple->replica_ptrs[i]);
			if (ditem->rank == left_dtuple->rank)
			{
				//self should be freed at last
				continue;
			}
#ifdef GJK_DEBUG
			printf("Release_src lcheck1= i=%d\n", i);
#endif
			FreeDataTuple(ditem);
		}
	}
	else
	{
#ifdef GJK_DEBUG
		printf("Release_src lcheck\n");
#endif
		FreeDataTuple(left_dtuple);
	}
	if (right_dtuple->op == RING_ALLGATHER)
	{
		int i = 0;
		for (i = 0; i < left_dtuple->ring_num; i++)
		{
			DataTuple* ditem = static_cast<DataTuple*>(right_dtuple->replica_ptrs[i]);
			if (ditem->rank == right_dtuple->rank)
			{
				//self should be freed at last
				continue;
			}
			FreeDataTuple(ditem);
		}
	}
	else
	{
#ifdef GJK_DEBUG
		printf("Release_src rcheck\n");
#endif
		FreeDataTuple(right_dtuple);
	}
#ifdef GJK_DEBUG
	printf("Release_src lrcheck\n");
#endif
	//free(trs);

}
void MyRing::OutPutTuple(void* dataTuple,  bool freeMem = true)
{

	DataTuple* dtp = static_cast<DataTuple*>(dataTuple);
	std::ostringstream ss;
	std::lock_guard<std::mutex> lock(out_mutex);
	size_t i, j;
	size_t len = dtp->data_num;
	ss << std::this_thread::get_id();
	std::string idstr = ss.str();
	printf("OutPUT(%s):\n", idstr.c_str());
	printf("Name: %s\t OP: %s\n", dtp->data_name, getOp(dtp->op).c_str());
	printf("%s Direction  toRight? %d   len %ld  counter = %d\n", idstr.c_str(), dtp->toRight, len, dtp->scatter_gather_counter );
	if (dtp->op == RING_ALLGATHER)
	{

		for (i = 0; i < dtp->ring_num; i++)
		{

			printf("%ld:\t", i);
			DataTuple* ditem = static_cast<DataTuple*>(dtp->replica_ptrs[i]);
			if (ditem->rank == dtp->rank)
			{
				//self should be freed at last
				continue;
			}
			/*
			switch (ditem->data_type)
			{
			case FLOAT32:
			{
				float* arr = static_cast<float*>(ditem->data);

				for ( j = 0; j < len; j++)
				{
					printf("%lf\t", arr[j] );
				}
				printf("\n");

				break;
			}
			default:
			{
				break;
			}
			}
			**/
			if (freeMem)
			{
				printf("Free FIN  %s  %d\n", ditem->data_name, ditem->rank );
				FreeDataTuple(ditem);

			}

		}
		if (freeMem)
		{
			FreeDataTuple(dtp);
		}


	}
	else
	{
		/*
		switch (dtp->data_type)
		{
		case FLOAT32:
		{

			float* arr = static_cast<float*>(dtp->data);

			for ( i = 0; i < len; i++)
			{
				printf("%lf\t", arr[i] );
			}
			printf("\n");

			break;
		}
		default:
		{
			break;
		}
		}
		**/
		if (freeMem)
		{
			printf("Free FIN2  %s\n", dtp->data_name );
			FreeDataTuple(dtp);

		}

	}

	std::this_thread::sleep_for(std::chrono::seconds(1));

}

void MyRing::ProcessStageData(void* local_data, void* recv_data, int cur_stage)
{
#ifdef GJK_DEBUG
	printf("Process cur_stage = %d\n", cur_stage );

#endif
	bool isScatter = isScatterStage(cur_stage);

	MergeData(local_data, recv_data, isScatter);

	DataTuple* dt = static_cast<DataTuple*>(local_data);
	//printf("After Merge scacount=%d\n", dt->scatter_gather_counter );

	if ( (dt->op == RING_ALLREDUCE && dt->scatter_gather_counter == this->stage_num - 1 )
	        || (dt->op == RING_BROADCAST && dt->scatter_gather_counter >=  0 )
	        || (dt->op == RING_ALLGATHER && dt->scatter_gather_counter ==  this->ring_num - 1 - 1 ) )
	{
		//ok
		std::ostringstream ss;
		ss << std::this_thread::get_id();
		std::string idstr = ss.str();
		//printf("OK-FIN(%s):\n", idstr.c_str());
		FinishedTuple(local_data);
		//OutPutTuple(local_data);
	}
	else
	{

		switch (dt->op)
		{
		case RING_ALLREDUCE:
		{
			dt->scatter_gather_counter = cur_stage + 1;
			//printf("Enqueue2\n");
			EnqueSendQ(dt);
			//printf("FIN-Enqueue2\n");
		}

		break;
		case RING_ALLGATHER:
		{
			dt->scatter_gather_counter = cur_stage + 1;
			/*
			if (cur_stage == 0)
			{
				//for my own data, only send once to the next neighbour
				EnqueSendQ(dt);
			}
			**/

		}
		break;
		case RING_BROADCAST:
		{
			dt->scatter_gather_counter = cur_stage + 1;
			/*
			if (dt->rank == dt->broadcast_rank)
			{
				//Actually, no one will come here except the broadcaster
				EnqueSendQ(dt);

			}
			**/
		}
		break;
		default:
			break;
		}

		//dt->scatter_gather_counter = cur_stage + 1;
		//move upward
		//process_queues[cur_statge + 1].push(local_data);
		if (dt->toRight)
		{
			process_queues_to_right[cur_stage + 1].push(local_data);
		}
		else
		{
			process_queues_to_left[cur_stage + 1].push(local_data);
		}
	}


}
void MyRing::BackGround2LeftThreadCallback()
{
	while (!shut_down)
	{
		{
#ifdef GJK_DEBUG
			//printf("BackGround2LeftThreadCallback\n");
#endif
			std::lock_guard<std::mutex> lock(new_queue_to_left_mutex);
			while (!new_queue_to_left.empty())
			{
				//printf("Enter new_queue_to_left\n");
				void* msg = new_queue_to_left.front();
				DataTuple* dt = static_cast<DataTuple*>(msg);
				dt->scatter_gather_counter = 0;
				//printf("put to sendQ\n");
				{
					//OutPutTrs();
				}
				if ( (!(dt->op == RING_BROADCAST && dt->rank != dt->broadcast_rank)))
				{
#ifdef GJK_DEBUG
					printf("dt rank  %d\n", dt->rank);
					assert(dt->rank >= 0 && dt->rank <= 3);
#endif
					EnqueSendQ(dt);
				}
#ifdef GJK_DEBUG
				else
				{
					//printf("No Emq\n");
				}
#endif
				//EnqueSendQ(dt);
				//process_queues[0].push(msg);
				process_queues_to_left[0].push(msg);
				new_queue_to_left.pop();
			}
		}
		int stage_id ;
		for ( stage_id = this->stage_num - 1; stage_id >= 0; stage_id--)
		{
			//printf("stage_id %d\n", stage_id );
			queue<pair<void*, void*>> result_qu;
			std::map<string, void*>::iterator vit;
			{
				std::lock_guard<std::mutex> lock(map_mtx_to_left);
				while (!process_queues_to_left[stage_id].empty())
				{
					void* msg = process_queues_to_left[stage_id].front();
					process_queues_to_left[stage_id].pop();

					DataTuple* dt  = static_cast<DataTuple*>(msg);

					//printf("dt =%p\n", dt );
					if (!(dt->toRight))
					{

						if (dt->op == RING_BROADCAST && dt->broadcast_rank == this->ring_rank)
						{
							//printf("OK-fin2\n");
							FinishedTuple(dt);
							//OutPutTuple(dt);
							continue;
						}

						//string kstr = dt->data_name;
						//string kstr = GenUniqueKey(dt);
						char full_name[header_name_len];
						sprintf(full_name, "%s_%d", dt->data_name, stage_id);
						string kstr = full_name;
						//printf("key str  %s map  %ld\n", kstr.c_str(), recv_buf_map.size());
						//printf("Left Finding %s\n", dt->data_name );
						vit = recv_buf_map_to_left.find(kstr);
						//std::cout << "find " << kstr << std::endl;

						if (vit != recv_buf_map_to_left.end())
						{
#ifdef GJK_DEBUG
							printf("Left Found %s\n", dt->data_name );
#endif
							pair<void*, void*> pitem = make_pair(msg, vit->second);
							result_qu.push(pitem);
							recv_buf_map_to_left.erase(vit);
							//printf("Erase one mapsize  %ld\n", recv_buf_map_to_left.size());
						}
						else
						{
							//printf("Left NOT Found %s\n", dt->data_name );
							void* nptr = NULL;
							pair<void*, void*> pitem = make_pair(msg, nptr);
							result_qu.push(pitem);
						}
					}



				}
				//printf("OK: empty  %d\n", process_queues[stage_id].empty() );
			}
			while (!result_qu.empty())
			{
				pair<void*, void*> pit = result_qu.front();
				if (pit.second == NULL)
				{
					//printf("Is here?  map_size=%ld  map\n", recv_buf_map_to_left.size() );
					process_queues_to_left[stage_id].push(pit.first);
				}
				else
				{
					//printf("OK ELE  %p  %p\n", pit.second, static_cast<DataTuple*>(pit.second)->data );
					//printf("Before Process stage_id = %d\n", stage_id );
					ProcessStageData(pit.first, pit.second, stage_id);
					//printf(" q1.size = %ld  q2.size=%ld\n", process_queues_to_left[0].size(), process_queues_to_left[1].size());
				}
				result_qu.pop();
			}
		}
		//std::this_thread::sleep_for(std::chrono::seconds(1));
		//getchar();
	}
	printf("Terminated BackGround2LeftThreadCallback\n");

}
void MyRing::BackGround2RightThreadCallback()
{
	while (!shut_down)
	{
#ifdef GJK_DEBUG
		//printf("BackGroundThreadCallback\n");
#endif
		//getchar();
		//printf("BackGround2RightThreadCallback\n");
		//send new data
		{

			std::lock_guard<std::mutex> lock(new_queue_to_right_mutex);
			while (!new_queue_to_right.empty())
			{
				void* msg = new_queue_to_right.front();
				DataTuple* dt = static_cast<DataTuple*>(msg);
				dt->scatter_gather_counter = 0;
				//printf("put to sendQ right\n");
				{
					//OutPutTrs();
				}
				if ( (!(dt->op == RING_BROADCAST && dt->rank != dt->broadcast_rank)))
				{
#ifdef GJK_DEBUG
					printf("Enque1-right\n");
#endif

					EnqueSendQ(dt);
					//printf("FIN-Enque1\n");
				}
#ifdef GJK_DEBUG
				else
				{
					//printf("No Emq\n");
				}
#endif
				process_queues_to_right[0].push(msg);
				new_queue_to_right.pop();
			}
		}

		int stage_id ;
		for ( stage_id = this->stage_num - 1; stage_id >= 0; stage_id--)
		{
			//printf("stage_id %d\n", stage_id );
			queue<pair<void*, void*>> result_qu;
			std::map<string, void*>::iterator vit;
			{
				std::lock_guard<std::mutex> lock(map_mtx_to_right);
				while (!process_queues_to_right[stage_id].empty())
				{
					void* msg = process_queues_to_right[stage_id].front();
					process_queues_to_right[stage_id].pop();
					DataTuple* dt  = static_cast<DataTuple*>(msg);

					if (dt->op == RING_BROADCAST && dt->broadcast_rank == this->ring_rank)
					{
						//printf("OK-Fin2\n");
						FinishedTuple(dt);
						//OutPutTuple(dt);
						continue;
					}


					//string kstr = dt->data_name;
					char full_name[header_name_len];
					sprintf(full_name, "%s_%d", dt->data_name, stage_id);
					string kstr = full_name;

					//printf("Right Finding %s\n", dt->data_name );
					vit = recv_buf_map_to_right.find(kstr);
					//std::cout << "find " << kstr << std::endl;
					if (vit != recv_buf_map_to_right.end())
					{

						//printf("Right FOUND %s \n", kstr.c_str() );
						pair<void*, void*> pitem = make_pair(msg, vit->second);
						//printf("OK FOUND  %p  %p\n", vit->second, static_cast<DataTuple*>(vit->second)->data );
						result_qu.push(pitem);
						//printf("OK PIT  %p  %p\n", pitem.second, static_cast<DataTuple*>(pitem.second)->data );
						recv_buf_map_to_right.erase(vit);
						//printf("Erase one mapsize  %ld\n", recv_buf_map_to_right.size());
					}
					else
					{
						//printf("Right Not FOUND %s\n", dt->data_name );
						void* nptr = NULL;
						pair<void*, void*> pitem = make_pair(msg, nptr);
						result_qu.push(pitem);
					}

				}
				//printf("OK: empty  %d\n", process_queues[stage_id].empty() );
			}
			while (!result_qu.empty())
			{
				pair<void*, void*> pit = result_qu.front();
				if (pit.second == NULL)
				{

					process_queues_to_right[stage_id].push(pit.first);
				}
				else
				{

					//printf("OK ELE  %p  %p\n", pit.second, static_cast<DataTuple*>(pit.second)->data );
					//printf("Before Process stage_id = %d\n", stage_id );
					ProcessStageData(pit.first, pit.second, stage_id);
					//printf(" q1.size = %ld  q2.size=%ld\n", process_queues_to_right[0].size(), process_queues_to_right[1].size());
				}
				result_qu.pop();
			}
		}
		//std::this_thread::sleep_for(std::chrono::seconds(1));
		//getchar();
	}
	printf("Terminated BackGround2RightThreadCallback\n");

}

void MyRing::Send2RightThreadCallback()
{
	//std::cerr << "Send2RightThreadCallback" << std::endl;
#ifdef GJK_DEBUG
	printf("Send2RightThreadCallback\n");
	//OutPutTrs();
#endif
	int right_idx = this->getRightNeighbour(this->ring_rank);
#ifdef GJK_DEBUG
	printf("local %s Connecting 2Right %s  %d\n", this->to_right_ip_arrs[this->ring_rank], this->to_right_ip_arrs[right_idx], this->listen_for_left_connection_port );
#endif
	int send_fd =  InitConnection(this->to_right_ip_arrs[this->ring_rank], this->to_right_ip_arrs[right_idx], this->listen_for_left_connection_port);
#ifdef GJK_DEBUG
	printf("local %s Connected 2Right %s  %d\n", this->to_right_ip_arrs[this->ring_rank], this->to_right_ip_arrs[right_idx], this->listen_for_left_connection_port );
#endif
	to_right_connected = true;
	while (!shut_down)
	{
		void* msg = NULL;
		//Data Name, scatter_gather_counter,  dataType, data-length, data
		{
			std::lock_guard<std::mutex>lock(right_queue_mtx);
			//if (this->to_right_queue_tail > this->to_right_queue_front)
			if (!to_right_queue.empty())
			{
				//int qu_idx = (this->to_right_queue_front) % QueueLen;
				//void* msg = this->to_right_queue[qu_idx];
				msg = to_right_queue.front();
				to_right_queue.pop();
			}
		}
		if (msg)
		{

			DataTuple* dtuple = static_cast<DataTuple*>(msg);
#ifdef GJK_DEBUG
			printf("Send2RightThreadCallback\n");
			assert(dtuple->rank <= 3 && dtuple->rank >= 0);
#endif
			//printf("before send op %d\n", dtuple->op );
			size_t len = sizeof(DataTuple) + (dtuple->data_num) * (sizeoftype(dtuple->data_type));

			//printf("Sending2Right len=%ld  %d  %s\n", len, this->from_left_port_arrs[right_idx], dtuple->data_name );

			//OutPutTuple(dtuple);
			//getchar();
			int nwt = write(send_fd, msg, len );
			if (nwt < 0)
			{
				printf("Send FAIL-RIGHT\n");
			}
			else
			{
				//printf("Sended2Right Finished  nwt=%d  name = %s\n", nwt, dtuple->data_name);
			}
			free(msg);

		}
		//printf("Finished Send 2 Right\n");
		//getchar();
		//std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	printf("Terminated  Send2RightThreadCallback\n");
}
void MyRing::Send2LeftThreadCallback()
{
	//std::cerr << "Send2LeftThreadCallback" << std::endl;
#ifdef GJK_DEBUG
	printf("Send2LeftThreadCallback\n");
	//OutPutTrs();
#endif
	int left_idx = this->getLeftNeighbour(this->ring_rank);
#ifdef GJK_DEBUG
	printf("left_idx=%d\n", left_idx );
	printf("local %s Connecting 2Left  %s  %d\n",  this->to_left_ip_arrs[this->ring_rank], this->to_left_ip_arrs[left_idx], this->listen_for_right_connection_port);
#endif
	int send_fd =  InitConnection(this->to_left_ip_arrs[this->ring_rank], this->to_left_ip_arrs[left_idx], this->listen_for_right_connection_port);
#ifdef GJK_DEBUG
	printf("local %s Connected 2Left  %s  %d\n",  this->to_left_ip_arrs[this->ring_rank], this->to_left_ip_arrs[left_idx], this->listen_for_right_connection_port);
#endif
	to_left_connected = true;
	while (!shut_down)
	{
		void* msg = NULL;
		{
			//printf("Dequeue\n");
			std::lock_guard<std::mutex>lock(left_queue_mtx);
			if (!to_left_queue.empty())
			{
				msg = to_left_queue.front();
				to_left_queue.pop();
			}
			else
			{
				//printf("Empty Left QUEU\n");
			}

		}

		if (msg)
		{
			DataTuple* dtuple = static_cast<DataTuple*>(msg);
			size_t len = sizeof(DataTuple) + (dtuple->data_num) * (sizeoftype(dtuple->data_type));
#ifdef GJK_DEBUG

			printf("Send2LefttThreadCallback\n");
			assert(dtuple->rank <= 3 && dtuple->rank >= 0);
#endif
			//OutPutTuple(dtuple);
			//getchar();
			int nwt = write(send_fd, msg, len );
			if (nwt < 0)
			{
				printf("Send FAIL-LEFT\n");
			}
#ifdef GJK_DEBUG
			else
			{
				printf("Sended2Left Finished  nwt=%d  name = %s\n", nwt, dtuple->data_name);
			}
#endif
			free(msg);
		}
		//printf("Finished Send 2 Left\n");
		//getchar();
		//std::this_thread::sleep_for(std::chrono::seconds(1));
		//Tensor recved  = this.from_left_queue.front();
	}
	printf("Terminated  Send2LeftThreadCallback\n");
}
int MyRing::Wait4Connection(int bind_port)
{
	int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
	struct sockaddr_in sin;
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET; //ipv4
	sin.sin_port = htons(bind_port);//server listen public ports
	sin.sin_addr.s_addr = INADDR_ANY;//listen any connects
	while (bind(listen_fd, (struct sockaddr*)&sin, sizeof(sin)) < 0)
	{
		//std::cerr << "error in bind socket" << std::endl;
		printf("Error in bind  %d\n", bind_port  );
		std::this_thread::sleep_for(std::chrono::seconds(5));
		//exit(0);
	}

	if (listen(listen_fd, 1) == -1)
	{
		std::cerr << "error in server listen..." << std::endl;
		exit(-1);
	}
	printf("Listening... %d\n", bind_port );
	//getchar();
	struct sockaddr_in client_addr;
	int connected_fd, addr_len;
	addr_len = sizeof(struct sockaddr_in);
	connected_fd = accept(listen_fd, (struct sockaddr*) & (client_addr), (socklen_t*)&addr_len);
	return connected_fd;
}
char* MyRing::RecvFixedData(int connected_fd, size_t len)
{

	char* data_ptr = (char*)malloc(len) ;
	int cur_len = 0;
	int remain_len = len;
	int ans_len = 0;
	while ( (!shut_down) && remain_len > 0)
	{
		ans_len = recv(connected_fd, data_ptr + cur_len, remain_len, 0 );

#ifdef GJK_DEBUG
		if (ans_len > 0)
			printf("len = %ld ans_len = %d\n", len, ans_len );
#endif
		if (ans_len > 0)
		{
			cur_len += ans_len;
			remain_len -= ans_len;
		}
		if (remain_len <= 0)
		{
			return data_ptr;
		}
		//std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	return NULL;
}
void MyRing::ProcessRecvData(int connected_fd)
{
	//printf("ProcessRecvData\n");
	//set non-block
	int flags = fcntl(connected_fd, F_GETFL, 0);
	flags |= O_NONBLOCK;
	fcntl(connected_fd, F_SETFL, flags);
	//int header_len = header_name_len + sizeof(int) + sizeof(RING_TYPE) + sizeof(int);
	int header_len = sizeof(DataTuple);
	char* header_msg = this->RecvFixedData(connected_fd, header_len);

	if (header_msg == NULL)
	{
		printf("header_msg NULL\n");
		//exit(1);
		return;
	}
	DataTuple* dtuple = static_cast<DataTuple*>( static_cast<void*>(header_msg) );


#ifdef GJK_DEBUG
	printf("dtuple op %d  rank %d\n", dtuple->op, dtuple->rank );
#endif
	//printf("Received Name  %s  rank %d\n", dtuple->data_name, dtuple->rank );
	int data_len = (dtuple->data_num) * this->sizeoftype(dtuple->data_type);
	//printf("data_num = %d   data_len = %d\n", dtuple->data_num, data_len );
	char* data_msg = this->RecvFixedData(connected_fd, data_len);
	dtuple->data = data_msg;

	char full_name[header_name_len];
	sprintf(full_name, "%s_%d", dtuple->data_name, dtuple->scatter_gather_counter);
	//string keyname = dtuple->data_name;
	string keyname = full_name;
	//got a complete data  block and insert to the map
	//should locked
	{



		if (dtuple->toRight)
		{
			std::lock_guard<std::mutex>lock(map_mtx_to_right);
#ifdef GJK_DEBUG
			printf("Before Insert Map %ld  %s\n", recv_buf_map_to_right.size(), keyname.c_str());
#endif

			{
				std::map<string, void*>::iterator vit = recv_buf_map_to_right.find(keyname);
				if (vit != recv_buf_map_to_right.end())
				{
					printf("Will Override!! %s\n", keyname.c_str());
				}
#ifdef GJK_DEBUG
				printf("dddd  %s   %d  %d  %d\n", dtuple->data_name, dtuple->rank, dtuple->scatter_gather_counter, dtuple->op);
#endif
				assert(dtuple->rank <= 3 && dtuple->rank >= 0);
				recv_buf_map_to_right.insert(make_pair(keyname, static_cast<void*>(dtuple)));
				//printf("Right Map2 Inserted  %s\n", keyname.c_str() );
				//getchar();
			}

#ifdef GJK_DEBUG
			printf("Insert Map %ld\n", recv_buf_map_to_right.size());
#endif
		}
		else
		{
			std::lock_guard<std::mutex>lock(map_mtx_to_left);
#ifdef GJK_DEBUG
			printf("Before Insert Map %ld  %s\n", recv_buf_map_to_left.size(), keyname.c_str());
#endif

			{
#ifdef GJK_DEBUG
				printf("llll  %s   %d  %d  %d\n", dtuple->data_name, dtuple->rank, dtuple->scatter_gather_counter, dtuple->op);
				assert(dtuple->rank <= 3 && dtuple->rank >= 0);
#endif
				recv_buf_map_to_left.insert(make_pair(keyname, static_cast<void*>(dtuple)));
#ifdef GJK_DEBUG
				printf("Left Map2 Inserted  %s\n", keyname.c_str() );
				//getchar();
#endif
			}

#ifdef GJK_DEBUG
			printf("Insert Map %ld\n", recv_buf_map_to_left.size());
#endif
		}

	}
	//lock will be released auytomatically
}
void MyRing::Recv4LeftThreadCallback()
{
	//std::cerr << "Recv4LeftThreadCallback" << std::endl;
#ifdef GJK_DEBUG
	printf("Recv4LeftThreadCallback\n");
	//OutPutTrs();
#endif
	int bind_port = this->listen_for_left_connection_port;
	int connected_fd = this->Wait4Connection(bind_port);
#ifdef GJK_DEBUG
	int left_recved = 0;
#endif
	while (!shut_down)
	{
#ifdef GJK_DEBUG
		printf("Recving Left Complete Msg\n");
#endif
		ProcessRecvData(connected_fd);
#ifdef GJK_DEBUG
		left_recved++;
		printf("Recved Left Complete Msg  %d\n", left_recved);
#endif
		//std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	printf("Terminated  Recv4LeftThreadCallback\n");

}
void MyRing::Recv4RightThreadCallback()
{
#ifdef GJK_DEBUG
	//std::cerr << "Recv4RightThreadCallback" << std::endl;
	printf("Recv4RightThreadCallback\n");
	//OutPutTrs();
#endif
	int bind_port = this->listen_for_right_connection_port;
	int connected_fd = this->Wait4Connection(bind_port);
#ifdef GJK_DEBUG
	int right_recvd = 0;
#endif
	while (!shut_down)
	{
#ifdef GJK_DEBUG
		printf("Recving Right Complete Msg\n");
#endif
		ProcessRecvData(connected_fd);
#ifdef GJK_DEBUG
		right_recvd++;
		printf("Recved Right Complete Msg %d\n", right_recvd);
#endif
		//std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	printf("Terminated  Recv4RightThreadCallback\n");

}
int MyRing::sizeoftype(RING_TYPE dt)
{
	/*
	switch (dt)
	{
	case FLOAT32:
		return 4;
	case FLOAT64:
		return 8;
	case LONGINT:
		return 8;
	case INTEGER:
		return 4;
	default:
		return -1;
	}
	**/
	return TYPE_SIZE[dt];
}

void* MyRing::GenAllReduceBuf(DataTuple* dtuple)
{
	int send_block_idx;
#ifdef GJK_DEBUG
	printf("R_CHeck1\n");
#endif
	if (dtuple->toRight)
	{

		send_block_idx = (2 * this->ring_num + this->ring_rank - dtuple->scatter_gather_counter) % (this->ring_num);
	}
	else
	{
		send_block_idx = (2 * this->ring_num + dtuple->scatter_gather_counter +  this->ring_rank) % (this->ring_num);
	}
#ifdef GJK_DEBUG
	printf("CHeck2 rn %d  rr  %d  sc  %d  tpr  %d  sb %d\n", this->ring_num, this->ring_rank, dtuple->scatter_gather_counter, dtuple->toRight, send_block_idx);
#endif
	int share_num = (dtuple->data_num) / (this->ring_num);
	int rest_num = (dtuple->data_num) % (this->ring_num);
	int temp_arr[this->ring_num];
	int p = 0;
#ifdef GJK_DEBUG
	printf("CHeck3\n");
#endif
	for (p = 0; p < this->ring_num; p++)
	{
		if (p < rest_num)
		{
			temp_arr[p] = share_num + 1;
		}
		else
		{
			temp_arr[p] = share_num;
		}
		//printf("temp p %d  %d\n", p, temp_arr[p] );
	}
#ifdef GJK_DEBUG
	printf("CHeck4\n");
#endif
	int start_idx = 0;
	for (p = 0; p < send_block_idx; p++)
	{
		start_idx += temp_arr[p];
	}
#ifdef GJK_DEBUG
	printf("CHeck5\n");
#endif
	int cnt = temp_arr[send_block_idx];
	RING_TYPE dtp = dtuple->data_type;
	char* tosend_buf = NULL;
	int unit_size = sizeoftype(dtp);
	//printf("EnQ \n");
	//float* fdata = (float*) dtuple->data;
	char* ch_data = (char*)dtuple->data;
#ifdef GJK_DEBUG
	printf("CHeck6  char  %d  sendblock %d  cnt=%d datype = %d, size=%ld\n", sizeof(char), send_block_idx, cnt, dtuple->data_type, sizeoftype(dtuple->data_type) );
#endif
///
	tosend_buf = (char*)malloc(sizeof(DataTuple) + sizeoftype(dtuple->data_type) * cnt);
#ifdef GJK_DEBUG
	size_t numOfe = sizeof(DataTuple) + sizeoftype(dtuple->data_type) * cnt;
	//tosend_buf = (char*)calloc(numOfe, sizeof(char));

	printf("numofe....%ld\n\n", numOfe);
#endif
	//tosend_buf = (char*)std::malloc(numOfe);
	//DataTuple* tupleheader = (DataTuple*)malloc(sizeof(DataTuple));

	//sprintf(tupleheader->data_name, "TensorScatter2Right%d-%d", this->scatter_gather_counter, this->ring_rank);
#ifdef GJK_DEBUG
	printf("CHeck7  char  %ld\n", sizeof(char));
#endif
	//memcpy(tupleheader, dtuple, sizeof(DataTuple));
	assert(tosend_buf != NULL);
	DataTuple* tupleheader = static_cast<DataTuple*>(static_cast<void*>(tosend_buf));
#ifdef GJK_DEBUG
	printf("CHeck7-5\n");
#endif
	/*
	for (size_t ss = 0; ss < numOfe; ss++ )
	{
		printf("%c", tosend_buf[ss]);
		if (ss % 100 == 0)
		{
			printf("|%d|", ss);
		}
	}
	printf("\n");
	**/

	memcpy(tosend_buf, dtuple, sizeof(DataTuple));
#ifdef GJK_DEBUG
	printf("CHeck8\n");
#endif
	//strcpy(tupleheader->data_name, dtuple->data_name);
#ifdef GJK_DEBUG
	printf("CHeck9\n");
#endif
	tupleheader->data_num = cnt;
	tupleheader->start_idx = start_idx;
	//tupleheader->scatter_gather_counter = dtuple->scatter_gather_counter;
	//tupleheader->rank = dtuple->rank;
	//tupleheader->scatter_gather_counter = dt->scatter_gather_counter; //unncecessary
#ifdef GJK_DEBUG
	printf("CHeck10\n");
#endif
	//memcpy(tosend_buf, tupleheader, sizeof(DataTuple));

#ifdef GJK_DEBUG
	printf("CHeck11 start_idx = %ld unit_size=%ld  cnt=%ld  dtsz = %ld\n", start_idx, unit_size, cnt,  sizeof(DataTuple));
#endif
	memcpy(tosend_buf + sizeof(DataTuple), ch_data + start_idx * unit_size, unit_size * cnt);
#ifdef GJK_DEBUG
	printf("CHeck12\n");
#endif
	return tosend_buf;

}
void* MyRing::GenAllGatherBuf(DataTuple* dtuple)
{
#ifdef GJK_DEBUG
	printf("G_CHECK1 rank %d\n", dtuple->rank);
#endif
	size_t sz = 0;
	sz += sizeof(DataTuple);
#ifdef GJK_DEBUG
	printf("G_CHECK2 rank %d\n", dtuple->rank);
#endif
	size_t data_sz = dtuple->data_num * sizeoftype(dtuple->data_type);
#ifdef GJK_DEBUG
	printf("G_CHECK3 rank %d\n", dtuple->rank);
#endif
	sz += data_sz;
#ifdef GJK_DEBUG
	printf("G_CHECK4 rank %d\n", dtuple->rank);
#endif
	char* tosend_buf = (char*)malloc(sz);
#ifdef GJK_DEBUG
	printf("G_CHECK5 rank %d\n", dtuple->rank);
#endif
	memcpy(tosend_buf, dtuple, sizeof(DataTuple));
#ifdef GJK_DEBUG
	printf("G_CHECK6 rank %d\n", dtuple->rank);
#endif
	memcpy(tosend_buf + sizeof(DataTuple), dtuple->data, data_sz);
#ifdef GJK_DEBUG
	printf("G_CHECK7 rank %d\n", dtuple->rank);
#endif
	return tosend_buf;

}
void* MyRing::GenBroadCastBuf(DataTuple* dtuple)
{

#ifdef GJK_DEBUG
	//printf("B_CHECK1\n");
#endif
	size_t sz = 0;
	sz += sizeof(DataTuple);
#ifdef GJK_DEBUG
	//printf("B_CHECK2\n");
#endif
	size_t data_sz = dtuple->data_num * sizeoftype(dtuple->data_type);
#ifdef GJK_DEBUG
	//printf("B_CHECK3\n");
#endif
	sz += data_sz;
#ifdef GJK_DEBUG
	//printf("B_CHECK4\n");
#endif
	char* tosend_buf = (char*)malloc(sz);
	assert(tosend_buf != NULL);
#ifdef GJK_DEBUG
	//printf("B_CHECK5\n");
#endif
	memcpy(tosend_buf, dtuple, sizeof(DataTuple));
#ifdef GJK_DEBUG
	//printf("B_CHECK6 name  %s rank  %d sca %d  dtuple  %p data  %p\n", dtuple->data_name, dtuple->rank, dtuple->scatter_gather_counter, dtuple, dtuple->data);
#endif
	memcpy(tosend_buf + sizeof(DataTuple), dtuple->data, data_sz); //Why
#ifdef GJK_DEBUG
	//printf("B_CHECK7\n");
#endif

	return tosend_buf;
}
void MyRing::EnqueSendQ(DataTuple* dtuple)
{
#ifdef GJK_DEBUG
	printf("EnqueuSendQ-1\n");
#endif
	void* tosend_buf = NULL;
#ifdef GJK_DEBUG
	printf("EnqueuSendQ-3  op  %d  dtuple  %p vrank  %d  broadcastrank = %d  data  %p\n",  dtuple->op, dtuple, dtuple->rank, dtuple->broadcast_rank, dtuple->data);
#endif
	switch (dtuple->op)
	{
	case RING_ALLREDUCE:
	{
		tosend_buf = GenAllReduceBuf(dtuple);
	}
	break;
	case RING_ALLGATHER:
	{
		tosend_buf = GenAllGatherBuf(dtuple);
	}
	break;
	case RING_BROADCAST:
	{
		tosend_buf = GenBroadCastBuf(dtuple);
	}
	break;
	}

	if (dtuple->toRight)
	{
#ifdef GJK_DEBUG
		printf("toright\n");
#endif
		{
			std::lock_guard<std::mutex>lock(right_queue_mtx);
			to_right_queue.push((void*)tosend_buf);
		}
#ifdef GJK_DEBUG
		printf("pending in right queue\n");
#endif
	}
	else
	{
#ifdef GJK_DEBUG
		printf("toleft\n");
		assert(dtuple->rank >= 0 && dtuple->rank <= 3);
		DataTuple* test = static_cast<DataTuple*>(static_cast<void*>(tosend_buf));
		assert(test->rank >= 0 && test->rank <= 3);
#endif
		{
			std::lock_guard<std::mutex>lock(left_queue_mtx);
			to_left_queue.push((void*)tosend_buf);
		}
	}
}

void MyRing::RingAllReduceMerge(DataTuple* recvTuple, DataTuple* localTuple, bool isScatter)
{
#ifdef GJK_DEBUG
	printf("Enter RingAllreduce\n");
#endif
	int s_idx = recvTuple->start_idx;
	//printf("s_idx =%d\n", s_idx);
	int data_num = recvTuple->data_num;
	//printf("data_num = %d\n", data_num );
	RING_TYPE dt = recvTuple->data_type;
	//printf("dt = %d\n", dt );
#ifdef GJK_DEBUG
	printf("s_idx %d  data_num  %d  counter = %d\n", s_idx, data_num, localTuple->scatter_gather_counter);
#endif
	switch (dt)
	{
	case T_FLOAT32:
	{
		float* recv_float_data = static_cast<float*>(recvTuple->data);
		float* my_data = static_cast<float*>(localTuple->data);
		int i = 0;


#ifdef GJK_DEBUG
		printf("Recved %d-%d ", localTuple->scatter_gather_counter, recvTuple->scatter_gather_counter);
		/*
		for (int pp = 0; pp < data_num; pp++)
		{
			printf("%lf\t", recv_float_data[pp]);
		}
		**/
		printf("\n");
#endif
		for (i = 0; i < data_num; i++)
		{

			if (isScatter)
			{
				my_data[s_idx + i] += recv_float_data[i];
			}
			else
			{
				my_data[s_idx + i] = recv_float_data[i];
			}

		}
		/*
				printf("After Merging...\n");
				for (i = 0; i < 8; i++)
				{

					printf("%lf\t", my_data[i] );

				}
				printf("\n");
		**/
		break;
	}
	default:
		break;
	}
	//getchar();
	//printf("Before Freeing recvTuple\n");
	FreeDataTuple(recvTuple);
	//printf("After FG\n");

}
void MyRing::RingAllGatherMerge(DataTuple* recvTuple, DataTuple* localTuple)
{
#ifdef GJK_DEBUG
	printf("AllGatherMeerge\n");
#endif
	int rank = recvTuple->rank;
#ifdef GJK_DEBUG
	printf("AllGatherMerge Check2, rank = %d\n", rank);
#endif
	localTuple->replica_ptrs[rank] = static_cast<void*>(recvTuple);
#ifdef GJK_DEBUG
	printf("AllGatherMerge Check3\n");
#endif
	if ( (!(recvTuple->toRight == true &&  getRightNeighbour(localTuple->rank) == recvTuple->rank) )
	        && (!(recvTuple->toRight == false &&  getLeftNeighbour(localTuple->rank) == recvTuple->rank) )
	   )
	{
#ifdef GJK_DEBUG
		printf("AllGatherMerge Check4\n");
#endif
		//printf("Before  scc  %d recv_rank  %d\n", recvTuple->scatter_gather_counter, recvTuple->rank );
		recvTuple->scatter_gather_counter++;
		//printf("After  %d\n", recvTuple->scatter_gather_counter );
		//getchar();
#ifdef GJK_DEBUG
		printf("AllGatherMerge Check5\n");
#endif
		EnqueSendQ(recvTuple);
#ifdef GJK_DEBUG
		printf("AllGatherMerge Check6\n");
#endif
	}
#ifdef GJK_DEBUG
	else
	{
		printf("Hit\n");
		printf("%d = %d\n", getRightNeighbour(localTuple->rank), recvTuple->rank);
	}
#endif
	return;
}
void MyRing::RingBroadCastMerge(DataTuple* recvTuple, DataTuple* localTuple)
{
	if (localTuple->data != NULL)
	{
		//printf("double %p\n", localTuple->data );
		free(localTuple->data);
	}
	localTuple->data = recvTuple->data;
#ifdef GJK_DEBUG
	printf("In RingBroadCast\n");

#endif
	//then the broadcast data should continue to be delivered to the next hop
	if ( (!(localTuple->toRight == true &&  getLeftNeighbour(localTuple->broadcast_rank) == localTuple->rank) )
	        && (!(localTuple->toRight == false &&  getRightNeighbour(localTuple->broadcast_rank) == localTuple->rank) )
	   )
	{
		//printf("Here Enque IT\n");
		EnqueSendQ(localTuple);
	}
	/*
		{
			char kchar[DATA_NAME_LEN];
			int len = strlen(localTuple->data_name);
			printf("len = %d\n", len );
			len -= 5;
			strncpy(kchar, localTuple->data_name, len);
			kchar[len] = '\0';
			string kstr = kchar;
			std::lock_guard<std::mutex> lock(trs_mutex);
			{
				printf("In BroadMerge\n");

				std::map<string, void*>::iterator vit = trs_map.find(kstr);
				assert(vit != trs_map.end());
				TensorRingStruct* ddd = static_cast<TensorRingStruct*>(vit->second);
				printf("Check in BroadMerege  name=%s  out1=%p\n", ddd->tensor_name.c_str(), ddd->output);
			}
		}
		**/
//!@!!!!!
	free(recvTuple);
}
void MyRing::MergeData(void* local_buf, void* recvbuf, bool isScatter)
{



	// add  sendQ OP
	DataTuple* recvTuple = static_cast<DataTuple*>(recvbuf);
	DataTuple* localTuple = static_cast<DataTuple*>(local_buf);
	/*
		{
			char kchar[DATA_NAME_LEN];
			int len = strlen(localTuple->data_name);
			printf("len = %d\n", len );
			len -= 5;
			strncpy(kchar, localTuple->data_name, len);
			kchar[len] = '\0';
			string kstr = kchar;
			std::lock_guard<std::mutex> lock(trs_mutex);
			{
				printf("In BroadMerge\n");

				std::map<string, void*>::iterator vit = trs_map.find(kstr);
				assert(vit != trs_map.end());
				TensorRingStruct* ddd = static_cast<TensorRingStruct*>(vit->second);
				printf("Check in MergeData  name=%s  out1=%p\n", ddd->tensor_name.c_str(), ddd->output);
			}
		}
		**/
#ifdef GJK_DEBUG
	printf("Merging data... name %s op  %d rank  %d  braodrank =%d\n", recvTuple->data_name, recvTuple->op, recvTuple->rank, recvTuple->broadcast_rank);
#endif
	switch (recvTuple->op)
	{
	case RING_ALLREDUCE:
		RingAllReduceMerge(recvTuple, localTuple, isScatter);
		break;
	case RING_ALLGATHER:
		RingAllGatherMerge(recvTuple, localTuple);
		break;
	case RING_BROADCAST:
		RingBroadCastMerge(recvTuple, localTuple);
		break;
	default:
#ifdef GJK_DEBUG
		printf("come to default\n");
#endif
		break;
	}




}

void MyRing::FreeDataTuple(DataTuple* dtuple)
{
	// non-recursive
	if (dtuple != NULL)
	{

		if (dtuple->data != NULL)
		{
			//printf("freeing  dtuple-data  %p  name=%s\n", dtuple->data, dtuple->data_name );
			free(dtuple->data);
			//printf("freed dtuple->data %p  name=%s\n", dtuple->data, dtuple->data_name);
			dtuple->data = NULL;
		}
		//printf("freeing dtuple %p   name  =%s\n", dtuple, dtuple->data_name );
		free(dtuple);
		//printf("freed dtuple   %p  name=%s\n", dtuple, dtuple->data_name );
		dtuple = NULL;
	}
}

void MyRing::ShutDown()
{
	shut_down = true;
	for (auto& tid : thread_vec)
	{
		tid.join();
	}
	std::this_thread::sleep_for(std::chrono::seconds(1));
}
MyRing::~MyRing()
{


}
