#include "MyRing.h"
//map<string, void*> MyRing::recv_buf_map;
//std::mutex MyRing:: mtx;
const int MyRing::header_name_len;
constexpr char* MyRing::ip_arrs[MAX_NUM];
constexpr int MyRing::from_left_port_arrs[MAX_NUM];
constexpr int MyRing::from_right_port_arrs[MAX_NUM];

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

//void MyRing::EnqueReduceQueue(int ele_num, bool toRight);
//std::queue<void*> MyRing::new_queue;
//std::mutex MyRing::new_queue_mtx;
//void* MyRing::to_right_queue[QueueLen];
//void* MyRing::to_left_queue[QueueLen];

MyRing::MyRing(int rn, int rr)
{
	//just for test
	this->ring_num = rn;
	this->ring_rank = rr % this->ring_num;

	this->stage_num = (this->ring_num - 1) * 2;
	//process_queues.resize(this->stage_num);
	process_queues_to_right.resize(this->stage_num);
	process_queues_to_left.resize(this->stage_num);

	printf("ring_num = %d rank = %d   from_left_port = %d  from_right_port=%d\n", this->ring_num, this->ring_rank, this->from_left_port_arrs[ring_rank], from_right_port_arrs[ring_rank] );

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
void MyRing::EnqueQueue(int id, int ele_num, bool toRight, RING_OP r_op)
{
	srand((unsigned)time(NULL));
	DataTuple* dtuple = (DataTuple*)malloc(sizeof(DataTuple));
	stringstream ss;
	sprintf(dtuple->data_name, "Tensor_%d", id);
#ifdef GJK_DEBUG
	printf("In Enqueue  Reduce: %s\n", dtuple->data_name);
#endif

	//printf("OutPUT(%s):\n", idstr.c_str());
	//strcpy(dtuple->data_name, );
	dtuple->start_idx = 0;
	dtuple->rank = ring_rank;
	dtuple->broadcast_rank = 0;
	dtuple->toRight = toRight;
	dtuple->data_type = FLOAT32;
	dtuple->data_num = ele_num;
	dtuple->op = r_op;
	dtuple->replica_ptrs.resize(ring_num);
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

}
void MyRing::bench_test(size_t thread_num)
{
	size_t i = 0;
	std::vector<std::thread> vec;
	for ( i = 0; i < thread_num * 3; i += 3)
	{
		srand((unsigned)time(NULL));
		int ele_num = 100;
		bool toRight = (i % 2 == 1);

		std::thread titem(MyRing::EnqueQueue, i, ele_num, toRight, RING_BROADCAST);
		///Eoor//
		vec.push_back(std::move(titem));

		std::thread titem2(MyRing::EnqueQueue, i + 1, ele_num, toRight, RING_ALLGATHER);
		vec.push_back(std::move(titem2));


		std::thread titem3(MyRing::EnqueQueue, i + 2 , ele_num, toRight, RING_ALLREDUCE);
		vec.push_back(std::move(titem3));

		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	for (i = 0; i < vec.size(); i++)
	{
		printf("Thread %ld started\n", i );
		vec[i].join();
	}
}
void MyRing::InitBGThread()
{
	bench_test(60);

	printf("Enque Success\n");

	std::thread td3(&MyRing::Recv4LeftThreadCallback, this);
	std::thread td4(&MyRing::Recv4RightThreadCallback, this);
	std::thread td1(&MyRing::Send2RightThreadCallback, this);
	std::thread td2(&MyRing::Send2LeftThreadCallback, this);
	//std::thread td5(&MyRing::BackGroundThreadCallback, this);
	std::thread td5(&MyRing::BackGround2RightThreadCallback, this);
	std::thread td6(&MyRing::BackGround2LeftThreadCallback, this);

	td1.join();
	td2.join();
	td3.join();
	td4.join();
	td5.join();
	td6.join();
}
int MyRing::InitConnection(char* ip_addr, int port)
{
	printf("InitConnection  %s  %d\n", ip_addr, port );
	struct sockaddr_in ser_in, local_in;/*server ip and local ip*/
	memset(&ser_in, 0, sizeof(ser_in));
	memset(&local_in, 0, sizeof(local_in));

	int tmp_skfd = socket(AF_INET, SOCK_STREAM, 0);/*local socket*/

	/*bind remote socket*/
	ser_in.sin_family = AF_INET;
	ser_in.sin_port = htons(port);/*connect to public port remote*/
	inet_pton(AF_INET, ip_addr, &ser_in.sin_addr);

	/*bind local part*/
	local_in.sin_family = AF_INET;

	//inet_pton(AF_INET, local_eth.c_str(), &local_in.sin_addr);
#ifdef GJK_DEBUG
	printf("ip = %s  port = %d\n", ip_addr, port );
#endif
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
			//std::cerr << "waiting to connect to server...." << ++connect_count << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));
			if (connect_count == -1)
			{
				std::cerr << "error in connect to server" << std::endl;
				//close(tmp_skfd);
				//exit(0);
			}

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
void MyRing::OutPutTuple(void* dataTuple)
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
		for (i = 0; i < dtp->replica_ptrs.size(); i++)
		{
			printf("%ld:\t", i);
			DataTuple* ditem = static_cast<DataTuple*>(dtp->replica_ptrs[i]);
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
		}

	}
	else
	{
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
	}


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
		printf("OK-FIN(%s):\n", idstr.c_str());
		OutPutTuple(local_data);
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
	while (1 == 1)
	{
		{
			//printf("BackGround2LeftThreadCallback\n");
			std::lock_guard<std::mutex> lock(new_queue_to_left_mutex);
			while (!new_queue_to_left.empty())
			{
				//printf("Enter \n");
				void* msg = new_queue_to_left.front();
				DataTuple* dt = static_cast<DataTuple*>(msg);
				dt->scatter_gather_counter = 0;
				//printf("put to sendQ\n");
				if ( (!(dt->op == RING_BROADCAST && dt->rank != dt->broadcast_rank)))
				{
					EnqueSendQ(dt);
				}
#ifdef GJK_DEBUG
				else
				{
					printf("No Emq\n");
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
							printf("OK-fin2\n");
							OutPutTuple(dt);
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


}
void MyRing::BackGround2RightThreadCallback()
{
	while (1 == 1)
	{
		//std::cerr << "BackGroundThreadCallback"  << std::endl;
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
				//printf("put to sendQ\n");

				if ( (!(dt->op == RING_BROADCAST && dt->rank != dt->broadcast_rank)))
				{
					//printf("Enque1\n");
					EnqueSendQ(dt);
					//printf("FIN-Enque1\n");
				}
#ifdef GJK_DEBUG
				else
				{
					printf("No Emq\n");
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
						printf("OK-fin2\n");
						OutPutTuple(dt);
						continue;
					}


					//string kstr = dt->data_name;
					char full_name[header_name_len];
					sprintf(full_name, "%s_%d", dt->data_name, stage_id);
					string kstr = full_name;
					//printf("key str  %s map  %ld\n", kstr.c_str(), recv_buf_map.size());
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
					//printf("Is here?  map_size=%ld  map\n", recv_buf_map.size() );
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


}

void MyRing::Send2RightThreadCallback()
{
	//std::cerr << "Send2RightThreadCallback" << std::endl;
#ifdef GJK_DEBUG
	printf("Send2RightThreadCallback\n");
#endif
	int right_idx = this->getRightNeighbour(this->ring_rank);
#ifdef GJK_DEBUG
	printf("Connecting 2Right %s  %d\n", this->ip_arrs[right_idx], this->from_left_port_arrs[right_idx] );
#endif
	int send_fd =  InitConnection(this->ip_arrs[right_idx], this->from_left_port_arrs[right_idx]);
#ifdef GJK_DEBUG
	printf("Connected 2Right %s  %d\n", this->ip_arrs[right_idx], this->from_left_port_arrs[right_idx] );
#endif
	while (1 == 1)
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
}
void MyRing::Send2LeftThreadCallback()
{
	//std::cerr << "Send2LeftThreadCallback" << std::endl;
#ifdef GJK_DEBUG
	printf("Send2LeftThreadCallback");
#endif
	int left_idx = this->getLeftNeighbour(this->ring_rank);
#ifdef GJK_DEBUG
	printf("left_idx=%d\n", left_idx );
	printf("Connecting 2Left  %s  %d\n", this->ip_arrs[left_idx], this->from_right_port_arrs[left_idx]);
#endif
	int send_fd =  InitConnection(this->ip_arrs[left_idx], this->from_right_port_arrs[left_idx]);
#ifdef GJK_DEBUG
	printf("Connected 2Left  %s  %d\n", this->ip_arrs[left_idx], this->from_right_port_arrs[left_idx]);
#endif
	while (1 == 1)
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
			printf("Sending2Left len=%ld  port=%d  %s\n", len, this->from_right_port_arrs[left_idx], dtuple->data_name);
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
	while (remain_len > 0)
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
	//set non-block
	int flags = fcntl(connected_fd, F_GETFL, 0);
	flags |= O_NONBLOCK;
	fcntl(connected_fd, F_SETFL, flags);
	//int header_len = header_name_len + sizeof(int) + sizeof(DataType) + sizeof(int);
	int header_len = sizeof(DataTuple);
	char* header_msg = this->RecvFixedData(connected_fd, header_len);

	if (header_msg == NULL)
	{
		printf("header_msg NULL\n");
		exit(1);
	}
	DataTuple* dtuple = static_cast<DataTuple*>( static_cast<void*>(header_msg) );


#ifdef GJK_DEBUG
	printf("dtuple op %d\n", dtuple->op );
#endif
	//printf("Received Name  %s  rank %d\n", dtuple->data_name, dtuple->rank );
	int data_len = (dtuple->data_num) * this->sizeoftype(dtuple->data_type);
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
#endif
	int bind_port = this->from_left_port_arrs[this->ring_rank];
	int connected_fd = this->Wait4Connection(bind_port);
#ifdef GJK_DEBUG
	int left_recved = 0;
#endif
	while (1 == 1)
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

}
void MyRing::Recv4RightThreadCallback()
{
#ifdef GJK_DEBUG
	//std::cerr << "Recv4RightThreadCallback" << std::endl;
	printf("Recv4RightThreadCallback\n");
#endif
	int bind_port = this->from_right_port_arrs[this->ring_rank];
	int connected_fd = this->Wait4Connection(bind_port);
#ifdef GJK_DEBUG
	int right_recvd = 0;
#endif
	while (1 == 1)
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

}
int MyRing::sizeoftype(DataType dt)
{
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
}

void* MyRing::GenAllReduceBuf(DataTuple* dtuple)
{
	int send_block_idx;
	if (dtuple->toRight)
	{
#ifdef GJK_DEBUG
		printf("ToRight EnSendQ\n");
#endif
		send_block_idx = (this->ring_num + this->ring_rank - dtuple->scatter_gather_counter) % (this->ring_num);
	}
	else
	{
#ifdef GJK_DEBUG
		printf("ToLeft EnSendQ\n");
#endif
		send_block_idx = (this->ring_num + dtuple->scatter_gather_counter +  this->ring_rank) % (this->ring_num);
	}

	int share_num = (dtuple->data_num) / (this->ring_num);
	int rest_num = (dtuple->data_num) % (this->ring_num);
	int temp_arr[this->ring_num];
	int p = 0;
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

	}
	int start_idx = 0;
	for (p = 0; p < send_block_idx; p++)
	{
		start_idx += temp_arr[p];
	}
#ifdef GJK_DEBUG
	printf("send_block_idx = %d  scatter_gather_counter = %d  start_idx = %d\n", send_block_idx, dtuple->scatter_gather_counter, start_idx );
#endif
	int cnt = temp_arr[send_block_idx];
	DataType dtp = dtuple->data_type;
	char* tosend_buf = NULL;
	int unit_size = sizeoftype(dtp);
	//printf("EnQ \n");
	//float* fdata = (float*) dtuple->data;
	char* ch_data = (char*)dtuple->data;

///
	tosend_buf = (char*)malloc(sizeof(DataTuple) + sizeoftype(dtuple->data_type) * cnt);
	DataTuple* tupleheader = (DataTuple*)malloc(sizeof(DataTuple));
	//sprintf(tupleheader->data_name, "TensorScatter2Right%d-%d", this->scatter_gather_counter, this->ring_rank);
	memcpy(tupleheader, dtuple, sizeof(DataTuple));
	strcpy(tupleheader->data_name, dtuple->data_name);
	tupleheader->data_num = cnt;
	tupleheader->start_idx = start_idx;
	tupleheader->scatter_gather_counter = dtuple->scatter_gather_counter;
	//tupleheader->scatter_gather_counter = dt->scatter_gather_counter; //unncecessary
	memcpy(tosend_buf, tupleheader, sizeof(DataTuple));

	memcpy(tosend_buf + sizeof(DataTuple), ch_data + start_idx * unit_size, unit_size * cnt);
#ifdef GJK_DEBUG
	char* te = ch_data + start_idx * unit_size;
	float* fda = static_cast<float*>(static_cast<void*>(te));
	printf("To send\t");
	for (int q = 0; q < cnt; q++)
	{
		printf("%lf\t", fda[q] );
	}
	printf("\n");
#endif
	return tosend_buf;

}
void* MyRing::GenAllGatherBuf(DataTuple* dtuple)
{

	size_t sz = 0;
	sz += sizeof(DataTuple);
	size_t data_sz = dtuple->data_num * sizeoftype(dtuple->data_type);
	sz += data_sz;
	char* tosend_buf = (char*)malloc(sz);
	memcpy(tosend_buf, dtuple, sizeof(DataTuple));
	memcpy(tosend_buf + sizeof(DataTuple), dtuple->data, data_sz);
	return tosend_buf;

}
void* MyRing::GenBroadCastBuf(DataTuple* dtuple)
{
	size_t sz = 0;
	sz += sizeof(DataTuple);
	size_t data_sz = dtuple->data_num * sizeoftype(dtuple->data_type);
	sz += data_sz;
	char* tosend_buf = (char*)malloc(sz);
	memcpy(tosend_buf, dtuple, sizeof(DataTuple));
	memcpy(tosend_buf + sizeof(DataTuple), dtuple->data, data_sz);
	return tosend_buf;
}
void MyRing::EnqueSendQ(DataTuple* dtuple)
{
	//printf("EnqueuSendQ\n");
	void* tosend_buf = NULL;
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
	DataType dt = recvTuple->data_type;
	//printf("dt = %d\n", dt );
#ifdef GJK_DEBUG
	printf("s_idx %d  data_num  %d  counter = %d\n", s_idx, data_num, localTuple->scatter_gather_counter);
#endif
	switch (dt)
	{
	case FLOAT32:
	{
		float* recv_float_data = static_cast<float*>(recvTuple->data);
		float* my_data = static_cast<float*>(localTuple->data);
		int i = 0;

		/*
				printf("Before Merging...\n");
				OutPutTuple(recvTuple);
				for (i = 0; i < 8; i++)
				{

					printf("%lf\t", my_data[i] );

				}
				printf("\n");
		**/
#ifdef GJK_DEBUG
		printf("Recved %d-%d ", localTuple->scatter_gather_counter, recvTuple->scatter_gather_counter);
		for (int pp = 0; pp < data_num; pp++)
		{
			printf("%lf\t", recv_float_data[pp]);
		}
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
	FreeDataTuple(recvTuple);

}
void MyRing::RingAllGatherMerge(DataTuple* recvTuple, DataTuple* localTuple)
{
	//printf("AllGatherMeerge\n");
	int rank = recvTuple->rank;
	localTuple->replica_ptrs[rank] = static_cast<void*>(recvTuple);
	if ( (!(recvTuple->toRight == true &&  getRightNeighbour(localTuple->rank) == recvTuple->rank) )
	        && (!(recvTuple->toRight == false &&  getLeftNeighbour(localTuple->rank) == recvTuple->rank) )
	   )
	{
		//printf("Before Enque2\n");
		//printf("Before  scc  %d recv_rank  %d\n", recvTuple->scatter_gather_counter, recvTuple->rank );
		recvTuple->scatter_gather_counter++;
		//printf("After  %d\n", recvTuple->scatter_gather_counter );
		//getchar();
		EnqueSendQ(recvTuple);
		//printf("After Enque2\n");
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
		free(localTuple->data);
	}
	localTuple->data = recvTuple->data;
#ifdef GJK_DEBUG
	printf("In RingBroadCast\n");
	OutPutTuple(localTuple);
#endif
	//then the broadcast data should continue to be delivered to the next hop
	if ( (!(localTuple->toRight == true &&  getLeftNeighbour(localTuple->broadcast_rank) == localTuple->rank) )
	        && (!(localTuple->toRight == false &&  getRightNeighbour(localTuple->broadcast_rank) == localTuple->rank) )
	   )
	{
		EnqueSendQ(localTuple);
	}

	free(recvTuple);
}
void MyRing::MergeData(void* local_buf, void* recvbuf, bool isScatter)
{

	// add  sendQ OP
	DataTuple* recvTuple = static_cast<DataTuple*>(recvbuf);
	DataTuple* localTuple = static_cast<DataTuple*>(local_buf);
	//printf("Merging data... op  %d \n", recvTuple->op);
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
			free(dtuple->data);
		}
		free(dtuple);
	}
}


MyRing::~MyRing()
{

}

