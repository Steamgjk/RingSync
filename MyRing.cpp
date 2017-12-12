#include "MyRing.h"
//map<string, void*> MyRing::recv_buf_map;
//std::mutex MyRing:: mtx;
const int MyRing::header_name_len;
constexpr char* MyRing::ip_arrs[MAX_NUM];
constexpr int MyRing::from_left_port_arrs[MAX_NUM];
constexpr int MyRing::from_right_port_arrs[MAX_NUM];

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
	//this->from_left_queue_front = 0;
	//this->from_left_queue_tail = 0;
	//this->to_right_queue_front = 0;
	//this->to_right_queue_tail = 0;
	//this->from_right_queue_front = 0;
	//this->from_right_queue_tail = 0;
	//this->to_left_queue_front = 0;
	//this->to_left_queue_tail = 0;

}



int MyRing::getLeftNeighbour()
{
	int rank = this->ring_rank - 1;
	while (rank < 0)
	{
		rank += this->ring_num;
	}
	return (rank) % (this->ring_num);
}
int MyRing::getRightNeighbour()
{
	int rank = this->ring_rank + 1;
	while (rank < 0)
	{
		rank += this->ring_num;
	}
	return (rank) % (this->ring_num);
}
void MyRing::InitBGThread()
{

	{
		DataTuple* _2right_dtuple = (DataTuple*)malloc(sizeof(DataTuple));
		strcpy(_2right_dtuple->data_name, "Test2Right");
		_2right_dtuple->start_idx = 0;
		_2right_dtuple->toRight = true;
		_2right_dtuple->data_type = FLOAT32;
		_2right_dtuple->data_num = 6;
		//
		float* fdata = (float*)malloc(sizeof(float) * (_2right_dtuple->data_num));
		int i = 0;
		for (i = 0; i < _2right_dtuple->data_num; i++ )
		{
			fdata[i] =  i * 0.1 + static_cast<float>(this->ring_rank);
		}
		_2right_dtuple->data = static_cast<void*>(fdata);
		//new_queue.push(_2right_dtuple);
		new_queue_to_right.push(_2right_dtuple);
	}


	{
		DataTuple*  _2left_dtuple  = (DataTuple*)malloc(sizeof(DataTuple));
		strcpy(_2left_dtuple->data_name, "Test2Left");
		_2left_dtuple->start_idx = 0;
		_2left_dtuple->toRight = false;
		_2left_dtuple->data_type = FLOAT32;
		_2left_dtuple->data_num = 6;
		//
		float* fdata = (float*)malloc(sizeof(float) * (_2left_dtuple->data_num));
		int i = 0;
		for (i = 0; i < _2left_dtuple->data_num; i++ )
		{
			fdata[i] = 0.5 + i * 0.1 + static_cast<float>(this->ring_rank);
			//fdata[i] = 0.5;
		}
		_2left_dtuple->data = static_cast<void*>(fdata);
		//new_queue.push(_2left_dtuple);
		new_queue_to_left.push(_2left_dtuple);
	}



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
	printf("ip = %s  port = %d\n", ip_addr, port );
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
void MyRing::OutPutTuple(void* dataTuple)
{
	DataTuple* dtp = static_cast<DataTuple*>(dataTuple);
	printf("Direction  toRight? %d\n", dtp->toRight );
	int i;
	int len = dtp->data_num;
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
void MyRing::ProcessStageData(void* local_data, void* recv_data, int cur_stage)
{
	printf("Process cur_stage = %d\n", cur_stage );
	{
		//printf("In Process %p  %p\n", recv_data, static_cast<DataTuple*>(recv_data)->data );
	}
	bool isScatter = isScatterStage(cur_stage);
	MergeData(local_data, recv_data, isScatter);
	if (cur_stage == this->stage_num - 1)
	{
		//ok
		printf("OK-FIN:\n");
		OutPutTuple(local_data);
	}
	else
	{
		DataTuple* dt = static_cast<DataTuple*>(local_data);
		dt->scatter_gather_counter = cur_stage + 1;
		//Put to sendQ
		EnqueSendQ(dt);
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
			std::lock_guard<std::mutex> lock(new_queue_to_left_mutex);
			while (!new_queue_to_left.empty())
			{
				//printf("Enter \n");
				void* msg = new_queue_to_left.front();
				DataTuple* dt = static_cast<DataTuple*>(msg);
				dt->scatter_gather_counter = 0;
				//printf("put to sendQ\n");
				EnqueSendQ(dt);
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
					DataTuple* dt  = static_cast<DataTuple*>(msg);
					printf("dt =%p\n", dt );
					if (!(dt->toRight))
					{
						string kstr = dt->data_name;
						//printf("key str  %s map  %ld\n", kstr.c_str(), recv_buf_map.size());
						vit = recv_buf_map_to_left.find(kstr);
						//std::cout << "find " << kstr << std::endl;
						if (vit != recv_buf_map_to_left.end())
						{
							pair<void*, void*> pitem = make_pair(msg, vit->second);
							result_qu.push(pitem);
							recv_buf_map_to_left.erase(vit);
							//printf("Erase one mapsize  %ld\n", recv_buf_map_to_left.size());
						}
						else
						{
							void* nptr = NULL;
							pair<void*, void*> pitem = make_pair(msg, nptr);
							result_qu.push(pitem);
						}
					}

					process_queues_to_left[stage_id].pop();
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
					printf("Before Process stage_id = %d\n", stage_id );
					ProcessStageData(pit.first, pit.second, stage_id);
					printf(" q1.size = %ld  q2.size=%ld\n", process_queues_to_left[0].size(), process_queues_to_left[1].size());
				}
				result_qu.pop();
			}
		}
		std::this_thread::sleep_for(std::chrono::seconds(1));
		//getchar();
	}


}
void MyRing::BackGround2RightThreadCallback()
{
	while (1 == 1)
	{
		//std::cerr << "BackGroundThreadCallback"  << std::endl;
		//getchar();
		//send new data
		{
			std::lock_guard<std::mutex> lock(new_queue_to_right_mutex);
			while (!new_queue_to_right.empty())
			{
				void* msg = new_queue_to_right.front();
				DataTuple* dt = static_cast<DataTuple*>(msg);
				dt->scatter_gather_counter = 0;
				//printf("put to sendQ\n");
				EnqueSendQ(dt);
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
					DataTuple* dt  = static_cast<DataTuple*>(msg);
					string kstr = dt->data_name;
					//printf("key str  %s map  %ld\n", kstr.c_str(), recv_buf_map.size());
					vit = recv_buf_map_to_right.find(kstr);
					//std::cout << "find " << kstr << std::endl;
					if (vit != recv_buf_map_to_right.end())
					{
						//printf("Ok FOUND\n");
						pair<void*, void*> pitem = make_pair(msg, vit->second);
						//printf("OK FOUND  %p  %p\n", vit->second, static_cast<DataTuple*>(vit->second)->data );
						result_qu.push(pitem);
						//printf("OK PIT  %p  %p\n", pitem.second, static_cast<DataTuple*>(pitem.second)->data );
						recv_buf_map_to_right.erase(vit);
						printf("Erase one mapsize  %ld\n", recv_buf_map_to_right.size());
					}
					else
					{
						//printf("Not FOUND\n");
						void* nptr = NULL;
						pair<void*, void*> pitem = make_pair(msg, nptr);
						result_qu.push(pitem);
					}
					process_queues_to_right[stage_id].pop();
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
					printf("Before Process stage_id = %d\n", stage_id );
					ProcessStageData(pit.first, pit.second, stage_id);
					printf(" q1.size = %ld  q2.size=%ld\n", process_queues_to_right[0].size(), process_queues_to_right[1].size());
				}
				result_qu.pop();
			}
		}
		std::this_thread::sleep_for(std::chrono::seconds(1));
		//getchar();
	}


}
/*
void MyRing::BackGroundThreadCallback()
{
	while (1 == 1)
	{
		//std::cerr << "BackGroundThreadCallback"  << std::endl;
		//getchar();
		//send new data
		{
			std::lock_guard<std::mutex> lock(new_queue_mtx);
			while (!new_queue.empty())
			{
				//printf("Enter \n");
				void* msg = new_queue.front();
				DataTuple* dt = static_cast<DataTuple*>(msg);
				dt->scatter_gather_counter = 0;
				//printf("put to sendQ\n");
				EnqueSendQ(dt);

				process_queues[0].push(msg);
				new_queue.pop();
			}
		}
		int stage_id ;
		for ( stage_id = this->stage_num - 1; stage_id >= 0; stage_id--)
		{
			//printf("stage_id %d\n", stage_id );
			queue<pair<void*, void*>> result_qu;
			std::map<string, void*>::iterator vit;
			{
				std::lock_guard<std::mutex> lock(mtx);


				while (!process_queues[stage_id].empty())
				{
					void* msg = process_queues[stage_id].front();
					DataTuple* dt  = static_cast<DataTuple*>(msg);
					string kstr = dt->data_name;
					//printf("key str  %s map  %ld\n", kstr.c_str(), recv_buf_map.size());
					vit = recv_buf_map.find(kstr);
					//std::cout << "find " << kstr << std::endl;
					if (vit != recv_buf_map.end())
					{
						//printf("Ok FOUND\n");
						pair<void*, void*> pitem = make_pair(msg, vit->second);
						//printf("OK FOUND  %p  %p\n", vit->second, static_cast<DataTuple*>(vit->second)->data );
						result_qu.push(pitem);
						//printf("OK PIT  %p  %p\n", pitem.second, static_cast<DataTuple*>(pitem.second)->data );
						recv_buf_map.erase(vit);
						printf("Erase one mapsize  %ld\n", recv_buf_map.size());
					}
					else
					{
						//printf("Not FOUND\n");
						void* nptr = NULL;
						pair<void*, void*> pitem = make_pair(msg, nptr);
						result_qu.push(pitem);
					}
					process_queues[stage_id].pop();
				}
				//printf("OK: empty  %d\n", process_queues[stage_id].empty() );
			}
			while (!result_qu.empty())
			{
				pair<void*, void*> pit = result_qu.front();
				if (pit.second == NULL)
				{
					//printf("Is here?  map_size=%ld  map\n", recv_buf_map.size() );
					process_queues[stage_id].push(pit.first);
				}
				else
				{
					//printf("OK ELE  %p  %p\n", pit.second, static_cast<DataTuple*>(pit.second)->data );
					printf("Before Process stage_id = %d\n", stage_id );
					ProcessStageData(pit.first, pit.second, stage_id);
					printf(" q1.size = %ld  q2.size=%ld\n", process_queues[0].size(), process_queues[1].size());
				}
				result_qu.pop();
			}
		}
		std::this_thread::sleep_for(std::chrono::seconds(1));
		//getchar();
	}


}
**/
void MyRing::Send2RightThreadCallback()
{
	//std::cerr << "Send2RightThreadCallback" << std::endl;
	printf("Send2RightThreadCallback\n");
	int right_idx = this->getRightNeighbour();
	printf("Connecting 2Right %s  %d\n", this->ip_arrs[right_idx], this->from_left_port_arrs[right_idx] );
	int send_fd =  InitConnection(this->ip_arrs[right_idx], this->from_left_port_arrs[right_idx]);
	printf("Connected 2Right %s  %d\n", this->ip_arrs[right_idx], this->from_left_port_arrs[right_idx] );
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
			size_t len = sizeof(DataTuple) + (dtuple->data_num) * (sizeoftype(dtuple->data_type));

			printf("Sending2Right len=%ld  %d\n", len, this->from_left_port_arrs[right_idx]);
			//OutPutTuple(dtuple);
			//getchar();
			int nwt = write(send_fd, msg, len );
			if (nwt < 0)
			{
				printf("Send FAIL-RIGHT\n");
			}
			free(msg);

		}
		//printf("Finished Send 2 Right\n");
		//getchar();
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}
void MyRing::Send2LeftThreadCallback()
{
	//std::cerr << "Send2LeftThreadCallback" << std::endl;
	printf("Send2LeftThreadCallback");
	int left_idx = this->getLeftNeighbour();
	printf("left_idx=%d\n", left_idx );
	printf("Connecting 2Left  %s  %d\n", this->ip_arrs[left_idx], this->from_right_port_arrs[left_idx]);
	int send_fd =  InitConnection(this->ip_arrs[left_idx], this->from_right_port_arrs[left_idx]);
	printf("Connected 2Left  %s  %d\n", this->ip_arrs[left_idx], this->from_right_port_arrs[left_idx]);
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
			printf("Sending2Left len=%ld  port=%d\n", len, this->from_right_port_arrs[left_idx]  );
			//OutPutTuple(dtuple);
			//getchar();
			int nwt = write(send_fd, msg, len );
			if (nwt < 0)
			{
				printf("Send FAIL-LEFT\n");
			}

			free(msg);
		}
		//printf("Finished Send 2 Left\n");
		//getchar();
		std::this_thread::sleep_for(std::chrono::seconds(1));
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
		std::this_thread::sleep_for(std::chrono::seconds(1));
		//exit(0);
	}

	if (listen(listen_fd, 1) == -1)
	{
		std::cerr << "error in server listen..." << std::endl;
		exit(-1);
	}
	printf("Listening... %d\n", bind_port );
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
		if (ans_len > 0)
			printf("len = %ld ans_len = %d\n", len, ans_len );
		if (ans_len > 0)
		{
			cur_len += ans_len;
			remain_len -= ans_len;
		}
		if (remain_len <= 0)
		{
			return data_ptr;
		}
		std::this_thread::sleep_for(std::chrono::seconds(1));
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

	int data_len = (dtuple->data_num) * this->sizeoftype(dtuple->data_type);
	char* data_msg = this->RecvFixedData(connected_fd, data_len);
	string keyname = dtuple->data_name;
	if (data_msg == NULL)
	{
		printf("data msg NULL\n");
		exit(1);
	}
	else
	{
		dtuple->data = data_msg;
	}
	//got a complete data  block and insert to the map
	//should locked
	{

		if (dtuple->toRight)
		{
			std::lock_guard<std::mutex>lock(map_mtx_to_right);
			printf("Before Insert Map %ld  %s\n", recv_buf_map_to_right.size(), keyname.c_str());
			recv_buf_map_to_right.insert(make_pair(keyname, static_cast<void*>(dtuple)));
			printf("Insert Map %ld\n", recv_buf_map_to_right.size());
		}
		else
		{
			std::lock_guard<std::mutex>lock(map_mtx_to_left);
			printf("Before Insert Map %ld  %s\n", recv_buf_map_to_left.size(), keyname.c_str());
			recv_buf_map_to_left.insert(make_pair(keyname, static_cast<void*>(dtuple)));
			printf("Insert Map %ld\n", recv_buf_map_to_left.size());
		}

		/*
		std::map<string, void*>::iterator vit = recv_buf_map.find(keyname);
		if (vit != recv_buf_map.end())
		{
			DataTuple* dd = static_cast<DataTuple*>(vit->second);

			float* fd = static_cast<float*>(dd->data);
			for (int i = 0; i < dd->data_num; i++)
			{
				printf("%lf\t", fd[i] );
			}
			printf("\n");
			printf("dd = %p  %p  datga = %p\n",  dd, vit->second, dd->data);
			getchar();

		}
		**/
	}
	//lock will be released auytomatically
}
void MyRing::Recv4LeftThreadCallback()
{
	//std::cerr << "Recv4LeftThreadCallback" << std::endl;
	printf("Recv4LeftThreadCallback\n");
	int bind_port = this->from_left_port_arrs[this->ring_rank];
	int connected_fd = this->Wait4Connection(bind_port);
	int left_recved = 0;
	while (1 == 1)
	{
		printf("Recving Left Complete Msg\n");
		ProcessRecvData(connected_fd);
		left_recved++;
		printf("Recved Left Complete Msg  %d\n", left_recved);
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

}
void MyRing::Recv4RightThreadCallback()
{
	//std::cerr << "Recv4RightThreadCallback" << std::endl;
	printf("Recv4RightThreadCallback\n");
	int bind_port = this->from_right_port_arrs[this->ring_rank];
	int connected_fd = this->Wait4Connection(bind_port);
	int right_recvd = 0;
	while (1 == 1)
	{
		printf("Recving Right Complete Msg\n");
		ProcessRecvData(connected_fd);
		right_recvd++;
		printf("Recved Right Complete Msg %d\n", right_recvd);
		std::this_thread::sleep_for(std::chrono::seconds(1));
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
void MyRing::EnqueSendQ(DataTuple* dt)
{
	int send_block_idx;
	if (dt->toRight)
	{
		printf("ToRight EnSendQ\n");
		send_block_idx = (this->ring_num + this->ring_rank - dt->scatter_gather_counter) % (this->ring_num);
	}
	else
	{
		printf("ToLeft EnSendQ\n");
		send_block_idx = (this->ring_num + dt->scatter_gather_counter +  this->ring_rank) % (this->ring_num);
	}

	printf("send_block_idx = %d  scatter_gather_counter = %d\n", send_block_idx, dt->scatter_gather_counter );
	int NumOfEle = dt->data_num;
	int share_num = (dt->data_num) / (this->ring_num);
	int start_idx = send_block_idx * share_num;
	bool toright = dt->toRight;
	DataType dtp = dt->data_type;
	char* tosend_buf = NULL;
	int unit_size = 0;
	//printf("EnQ \n");
	switch (dtp)
	{
	case FLOAT32:
	{
		unit_size = this->sizeoftype(dtp);
		float* fdata = (float*) dt->data;
		int cnt = share_num;
		if (send_block_idx == this->ring_num - 1)
		{
			//the last block
			cnt = NumOfEle - (share_num * (this->ring_num - 1) );
		}

		tosend_buf = (char*)malloc(sizeof(DataTuple) + sizeof(float) * cnt);
		DataTuple* tupleheader = (DataTuple*)malloc(sizeof(DataTuple));
		//sprintf(tupleheader->data_name, "TensorScatter2Right%d-%d", this->scatter_gather_counter, this->ring_rank);
		strcpy(tupleheader->data_name, dt->data_name);
		tupleheader->toRight = toright;
		tupleheader->data_num = cnt;
		tupleheader->data_type = dtp;
		tupleheader->start_idx = start_idx;
		//tupleheader->scatter_gather_counter = dt->scatter_gather_counter; //unncecessary
		memcpy(tosend_buf, tupleheader, sizeof(DataTuple));
		memcpy(tosend_buf + sizeof(DataTuple), fdata + start_idx, unit_size * cnt);
		break;

	}
	default:
		break;
	}
	if (toright)
	{
		printf("toright\n");

		{
			std::lock_guard<std::mutex>lock(right_queue_mtx);
			to_right_queue.push((void*)tosend_buf);
		}
	}
	else
	{
		printf("toleft\n");
		{
			std::lock_guard<std::mutex>lock(left_queue_mtx);
			to_left_queue.push((void*)tosend_buf);
		}
	}
}

void MyRing::MergeData(void* local_buf, void* recvbuf, bool isScatter)
{
	printf("Merging data... isScatter  %d\n", isScatter);
	// add
	DataTuple* recvTuple = static_cast<DataTuple*>(recvbuf);
	DataTuple* localTuple = static_cast<DataTuple*>(local_buf);
	/*
		if (local_buf != NULL)
		{
			printf("local_buf not null\n");
		}
		else
		{
			printf("local_buf null");
		}
		if (recvbuf != NULL)
		{
			printf("recvbuf not null\n");
		}
		else
		{
			printf("recvbuf null\n");
		}
		**/
	int s_idx = recvTuple->start_idx;
	//printf("s_idx =%d\n", s_idx);
	int data_num = recvTuple->data_num;
	//printf("data_num = %d\n", data_num );
	DataType dt = recvTuple->data_type;
	//printf("dt = %d\n", dt );
	printf("s_idx %d  data_num  %d\n", s_idx, data_num);
	switch (dt)
	{
	case FLOAT32:
	{
		float* recv_float_data = static_cast<float*>(recvTuple->data);
		float* my_data = static_cast<float*>(localTuple->data);
		int i = 0;
		/*
		printf("Coming hehre   dd %p  data %p\n", recvTuple, recvTuple->data);
		for (i = 0; i < data_num; i++)
		{
			printf("%lf\t", my_data[i]);
		}
		printf("\n");
		**/
		printf("Before Merging...\n");
		for (i = 0; i < 6; i++)
		{

			printf("%lf\t", my_data[i] );

		}
		printf("\n");
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
		printf("After Merging...\n");
		for (i = 0; i < 6; i++)
		{

			printf("%lf\t", my_data[i] );

		}
		printf("\n");
		break;
	}
	default:
		break;
	}

	if (recvbuf != NULL)
	{
		if (recvTuple->data != NULL)
		{
			free(recvTuple->data);
			recvTuple->data = NULL;
		}
		free(recvbuf);
		recvbuf = NULL;
	}

}



MyRing::~MyRing()
{

}

