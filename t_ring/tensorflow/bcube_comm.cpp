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
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
#include <unordered_map>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include "bcube_utils.h"
#include "bcube_ops.h"
#include "bcube_comm.h"
#include "bcube_message.h"

#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>



/*flag indicate background thread status.*/
static std::atomic_bool server_establisted(false);
static std::atomic_bool client_establisted(false);
//static void show_msg(void* row_data);

/* count all nodes N=n^k */
static void node_counts(bcube_struct& bcube_s)
{
	bcube_s.bcube_node_count = 1;
	for (int lev = 0; lev < bcube_s.bcube_level; lev++)
	{
		bcube_s.bcube_node_count *= bcube_s.bcube0_size;
	}
}

/*È·¶¨µ±Ç°bcubeËù´¦µÄ½ÚµãºÍÐÅÏ¢*/
static void setup_node(bcube_struct& bcube_s)
{
	char* __rank__ = getenv("BCUBE_RANK");
	if (__rank__ != NULL)bcube_s.rank = atoi(__rank__);
	else
	{
		std::cerr << "error in get env rank, you must set up it before run." << std::endl;
		exit(-1);
	}
	bcube_s.local_info.node_index = bcube_s.rank;
	for (size_t lev = 0; lev < bcube_s.topo.size(); lev++)/*add myself ip into mynodes*/
		bcube_s.local_info.myip.push_back(bcube_s.topo[lev][bcube_s.rank].ip);

	/*·¢ÏÖÁÚ¾Ó½Úµã£¬¼ÓÈëµ½neighbor_infoÖÐ*/
	for (int lev = 0; lev < bcube_s.bcube_level; lev++)/*each level*/
	{
		int * tmp_neigh = new int[bcube_s.bcube0_size - 1];
		std::vector<node> grp;
		Utils::getOneHopNeighbour(bcube_s.rank, lev, bcube_s.bcube0_size, 1, tmp_neigh);
		for (int neigh_index = 0; neigh_index < bcube_s.bcube0_size - 1; neigh_index++)
			grp.push_back(bcube_s.topo[lev][tmp_neigh[neigh_index]]);
		bcube_s.neighbor_info.push_back(grp);
		grp.clear();
		delete[] tmp_neigh;
	}
	return;

}

/*¼ÓÔØËùÓÐµÄÍøÂç½Úµã*/
/*
192.168.10.XXX
192.168.11.XXX
*/
static void topology_init(bcube_struct& bcube_s)
{
	printf("constructing a BCube(%d,%d) topology\n",bcube_s.bcube0_size,bcube_s.bcube_level);
	node_counts(bcube_s);
	for (int leve = 0; leve < bcube_s.bcube_level; leve++)
	{
		std::string ip_addr = "192.168.";
		std::string leve_str = std::to_string((leve + 10));
		std::vector<node> tp;/*each level*/
		node tmp_node;
		ip_addr += leve_str + std::string(".");

		for (int nodenum = 0; nodenum < bcube_s.bcube_node_count; nodenum++)
		{
			tmp_node.ip = ip_addr + std::to_string(nodenum + 10);
			tmp_node.node_index = nodenum;
			tp.push_back(tmp_node);
		}
		bcube_s.topo.push_back(tp);
	}
	printf("BCube(%d,%d) is constructed done!\n",bcube_s.bcube0_size,bcube_s.bcube_level);
}
struct bcube_global_struct;
static void insert_to_recv_queue(bcube_global_struct& bgs, received_tensor_entry& rs_e)
{
	auto& bs = bgs.bcube_s;
	auto& recv_queue = bgs.receiv_tmp_tensor;
	string name = rs_e.tensor_name;/*point to get tensor_name*/
	auto it = recv_queue.find(name);
	if (it != recv_queue.end())/*exist before and insert into.*/
	{
		//printf("%s exit before, append it behinds, it size is%d\n",name.c_str(),it->second.size());
		auto& vec_msg = it->second;
		vec_msg.push_back(std::move(rs_e));
		if (vec_msg.size() == (size_t)(bs.bcube0_size - 1)*bs.bcube_level)
		{/*if all received done, move tensor to received tensor.*/
			//printf("tensor %s is ready to reduce, move to received tensor buf.\n",it->first.c_str());
			{
				std::lock_guard<std::mutex> recv_lock(bgs.bcube_mutex);
				bgs.receiv_tensor.emplace(std::make_pair(name, std::move(vec_msg)));
			}
			recv_queue.erase(it);
		}
	}
	else
	{
		//printf("%s not exist... create one...\n",name.c_str());		
		std::vector<received_tensor_entry> msg_record;
		//std::this_thread::sleep_for(std::chrono::seconds(1));
		msg_record.push_back(std::move(rs_e));
		recv_queue.emplace(std::make_pair(name, std::move(msg_record)));
		//printf("insert receive done\n");
	}
	return;
}
extern void show_msg(void*);
void recv_loops(bcube_global_struct& bgs)
{
	bcube_struct& bs = bgs.bcube_s;
	int client_counts = (bs.bcube0_size - 1) * bs.bcube_level;
	printf("server is inited done, waiting for %d client connecting....:)\n",client_counts);
	while (client_counts-- > 0)
	{
		struct sockaddr_in client_addr;
		int connected_fd, addr_len;
		addr_len = sizeof(struct sockaddr_in);
		connected_fd = accept(bs.server_fd, (struct sockaddr*) & (client_addr), (socklen_t*)&addr_len);
		if (connected_fd == -1)
		{
			std::cerr << "error in server accept..." << std::endl;
			exit(-1);
		}
		fcntl(connected_fd, F_SETFL, fcntl(connected_fd, F_GETFL, 0) | O_NONBLOCK);
		bs.recv_fd.push_back(connected_fd);
		printf("client[%s,%d] is connecting now... \n", inet_ntoa(client_addr.sin_addr),client_addr.sin_port);
	}
	printf("%d clients have connected to my node, ready to receiving loops\n", client_counts);

	int msg_len = sizeof(msg_struct);
	auto& fd_vect = bgs.bcube_s.recv_fd;
	int fd_num = fd_vect.size();
	server_establisted = true;
	msg_struct msg_buf;
	while (true)
	{
		//printf("in recevie loops .................... inited.................\n");
		for (int fd_index = 0; fd_index < fd_num; fd_index++)
		{
			memset((void*)(&msg_buf), 0, msg_len);
			if (recv(fd_vect[fd_index], &msg_buf, msg_len, MSG_PEEK) != msg_len)continue;
			else
			{
				//printf("----------------------------------------------receive msg..................\n");
				assert((msg_buf.msg_length >= msg_len) && (msg_buf.data[0] == ','));
				void* new_msg = (void*)std::malloc(msg_buf.msg_length);
				assert(new_msg != nullptr);
				//printf("in receive msg loops: malloc %p\n", new_msg);
				memset(new_msg, 0, msg_buf.msg_length);
				if (recv(fd_vect[fd_index], new_msg, msg_buf.msg_length, MSG_PEEK) != msg_buf.msg_length)
				{
					;//printf("receive buffer is not ready, continue...\n");
				}
				else
				{
					assert(recv(fd_vect[fd_index], new_msg, msg_buf.msg_length, 0) == msg_buf.msg_length);
					received_tensor_entry e;
					show_msg(new_msg);
					tensor_msg::decode(e, new_msg);
					insert_to_recv_queue(bgs, e);
				}
				std::free(new_msg);
				//printf("in receive loops: free %p\n", new_msg);
				new_msg = nullptr;
			}
		}
	}
	return;
}

extern bcube_global_struct bcube_gs;

static void server_init(bcube_struct& bs)
{
	int init_loops=0;
	struct sockaddr_in sin;
	printf("init a server....\n");
	/*alloc a socket_fd and init*/
	bs.server_fd = socket(AF_INET, SOCK_STREAM, 0);
	assert(bs.server_fd > 0);
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;/*ipv4*/
	sin.sin_port = htons(bs.server_port);/*server listen public ports*/
	sin.sin_addr.s_addr = INADDR_ANY;/*listen any connects*/

	while(bind(bs.server_fd, (struct sockaddr*)&sin, sizeof(sin)) < 0)
	{
		std::cerr << "server init failed: error in bind socket, will try it again in 2 seconds..." << std::endl;
		if(init_loops>10) 
		{
			close(bs.server_fd);
			exit(-1);
		}
		std::this_thread::sleep_for(std::chrono::seconds(2));
		init_loops++;
	}

	int client_counts = (bs.bcube0_size - 1) * bs.bcube_level;
	if (listen(bs.server_fd, client_counts) == -1)
	{
		std::cerr << "server init failed: error in server listening" << std::endl;
		close(bs.server_fd);
		exit(-1);
	}


	bcube_gs.bg_thread.push_back(std::thread(recv_loops, std::ref(bcube_gs)));
	std::this_thread::sleep_for(std::chrono::seconds(1));
	return;
}



static void client_init(bcube_struct& bs)
{
	for (size_t lev = 0; lev < bs.neighbor_info.size(); lev++)
	{
		for (size_t index = 0; index < bs.neighbor_info[lev].size(); index++)
		{
			std::string local_eth = bs.local_info.myip[lev];/*get each lev ip*/
			struct sockaddr_in ser_in, local_in;/*server ip and local ip*/
			memset(&ser_in, 0, sizeof(ser_in));
			memset(&local_in, 0, sizeof(local_in));

			int tmp_skfd = socket(AF_INET, SOCK_STREAM, 0);/*local socket*/
			assert(tmp_skfd > 0);

			/*bind remote socket*/
			ser_in.sin_family = AF_INET;
			ser_in.sin_port = htons(bs.server_port);/*connect to public port remote*/
			inet_pton(AF_INET, bs.neighbor_info[lev][index].ip.c_str(), &ser_in.sin_addr);

			/*bind local part*/
			local_in.sin_family = AF_INET;
			std::cout << tmp_skfd << "  " << bs.neighbor_info[lev][index].node_index << " " << local_eth.c_str() << "----->" << bs.neighbor_info[lev][index].ip.c_str() << std::endl;
			inet_pton(AF_INET, local_eth.c_str(), &local_in.sin_addr);
			if (bind(tmp_skfd, (struct sockaddr*) & (local_in), sizeof(local_in)) != 0)
			{
				std::cerr << "error in bind local addr, client init error..." << std::endl;
				exit(-1);
			}

			if (1)/*connect to server*/
			{
				int connect_count = 0;
				while (connect(tmp_skfd, (struct sockaddr*) & (ser_in), sizeof(ser_in)) != 0)
				{
					//std::cout << "waiting to connect to server...." <<++connect_count<< std::endl;
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
					connect_count++;
					if (connect_count > 100 * 600)/*after 600 seconds, it will exit.*/
					{
						std::cerr <<600<< "seconds is passed, error in connect to server"<<bs.neighbor_info[lev][index].ip<<", check your network condition" << std::endl;
						close(tmp_skfd);
						exit(0);
					}
				}
				std::cout<< local_eth << " has connected to server[ " << bs.neighbor_info[lev][index].ip<< " , "<< bs.server_port << " ]" << std::endl;
			}
			bs.topo[lev][bs.neighbor_info[lev][index].node_index].remote_fd = tmp_skfd;
			bs.neighbor_info[lev][index].remote_fd = tmp_skfd;
		}
	}
	client_establisted = true;
	std::cout << "client inited done" << std::endl;
}

void show_each_node(bcube_struct& bs, int n)
{
	if (n < 0 || n > 8)
	{
		for (size_t node_index = 0; node_index < bs.nodes_send_strategy.size(); node_index++)
		{
			printf("node%lu:\n", node_index);
			for (size_t step_index = 0; step_index < bs.nodes_send_strategy[node_index].size(); step_index++)
			{
				printf("\tstep%lu:\n", step_index);
				for (size_t proc_index = 0; proc_index < bs.nodes_send_strategy[node_index][step_index].size(); proc_index++)
				{
					printf("\t\tprocess%lu:\n", proc_index);
					for (size_t send_index = 0; send_index < bs.nodes_send_strategy[node_index][step_index][proc_index].size(); send_index++)
					{
						auto& sends = bs.nodes_send_strategy[node_index][step_index][proc_index];
						printf("\t\t\tnode%lu--->node%d,with sock:%d, para:", node_index,
							sends[send_index].node_id, sends[send_index].socket_fd);
						for (size_t para_id = 0; para_id < sends[send_index].paraid.size(); para_id++)
							printf(" %d ", sends[send_index].paraid[para_id]);
						printf("\n");
					}
				}
			}
		}
	}
	else
	{
		int node_index = n;
		printf("node%d:\n", node_index);
		for (size_t step_index = 0; step_index < bs.nodes_send_strategy[node_index].size(); step_index++)
		{
			printf("\tstep%lu:\n", step_index);
			for (size_t proc_index = 0; proc_index < bs.nodes_send_strategy[node_index][step_index].size(); proc_index++)
			{
				printf("\t\tprocess%lu:\n", proc_index);
				for (size_t send_index = 0; send_index < bs.nodes_send_strategy[node_index][step_index][proc_index].size(); send_index++)
				{
					auto& sends = bs.nodes_send_strategy[node_index][step_index][proc_index];
					printf("\t\t\tnode%d--->node%d, with sockfd:%d, para:", node_index,
						sends[send_index].node_id, sends[send_index].socket_fd);
					for (size_t para_id = 0; para_id < sends[send_index].paraid.size(); para_id++)
						printf(" %d ", sends[send_index].paraid[para_id]);
					printf("\n");
				}
			}
		}
	}
}


void outputs(bcube_struct& bs, int** sendMatrix, int N, int para_num, int p, int s)
{
#if __outs_all_
	printf("para_num = %d\n", para_num);
#endif
	for (int i = 0; i < N; i++)
	{
		auto& strategy = bs.nodes_send_strategy[i][s][p];
		for (int j = 0; j < N; j++)
		{
			if (-1 != sendMatrix[i][j])
			{
				send_to_one to_one_node;
				to_one_node.node_id = j;
#if __outs_all_
				printf("\nnode%d->node%d, send paraID: %d", i, j, sendMatrix[i][j]);
				for (int cou = 1; cou < para_num; cou++)printf(", %d", sendMatrix[i][j] + cou);
#endif
				for (int col = 0; col < para_num; col++)to_one_node.paraid.push_back(sendMatrix[i][j] + col);
				to_one_node.block_num = para_num;
				strategy.push_back(to_one_node);
			}

		}
#if __outs_all_
		printf("\n");
#endif
	}
}
static void set_sock_fd(bcube_struct& bs)
{
	for (size_t step_index = 0; step_index < bs.my_strategy.size(); step_index++)
	{
		for (size_t proc_index = 0; proc_index < bs.my_strategy[step_index].size(); proc_index++)
		{
			for (size_t node_index = 0; node_index < bs.my_strategy[step_index][proc_index].size(); node_index++)
			{
				auto& each_node = bs.my_strategy[step_index][proc_index][node_index];
				for (size_t lev_index = 0; lev_index < bs.neighbor_info.size(); lev_index++)
				{
					for (size_t neigh_index = 0; neigh_index < bs.neighbor_info[lev_index].size(); neigh_index++)
					{
						if (each_node.node_id == bs.neighbor_info[lev_index][neigh_index].node_index)
							each_node.socket_fd = bs.neighbor_info[lev_index][neigh_index].remote_fd;
					}
				}
			}
		}
	}
}

static void get_send_strategy(bcube_struct& bs)
{
	int n = bs.bcube0_size;
	int k = bs.bcube_level - 1;
	int N = bs.bcube_node_count;
	int para_num;
	int **sendMatrix = new int*[N];

	for (int i = 0; i < N; i++)sendMatrix[i] = new int[N];

	{
		/*resize strategy vector*/
		bs.nodes_send_strategy.resize(N);
		for (size_t node_index = 0; node_index < bs.nodes_send_strategy.size(); node_index++)
		{
			bs.nodes_send_strategy[node_index].resize((k + 1) * 2);
			for (size_t step_index = 0; step_index < bs.nodes_send_strategy[node_index].size(); step_index++)
			{
				bs.nodes_send_strategy[node_index][step_index].resize((k + 1));
			}
		}
	}
	for (int s = 0; s <= k; s++)
	{
		for (int p = 0; p <= k; p++)
		{
#if __outs_all_
			printf("\n\nin scatter stage: %d\t using p %d\n", s, p);
#endif
			Utils::getScatterMatrix(p, s, n, k, N, sendMatrix, para_num);
			outputs(bs, sendMatrix, N, para_num, p, s);
		}

	}
	for (int s = 0; s <= k; s++)
	{
		for (int p = 0; p <= k; p++)
		{
#if __outs_all_
			printf("\n\nin gather stage: %d\t using p %d\n", s, p);
#endif
			Utils::getGatherMatrix(p, s, n, k, N, sendMatrix, para_num);
			outputs(bs, sendMatrix, N, para_num, p, s + k + 1);
		}
	}
	for (int row = 0; row < N; row++)
		delete[] sendMatrix[row];
	delete[] sendMatrix;
	bs.my_strategy = bs.nodes_send_strategy[bs.rank];
	set_sock_fd(bs);
	bs.nodes_send_strategy[bs.rank] = bs.my_strategy;
	//show_each_node(bs, bs.rank);
}


static bool check_bcube_is_inited_done(bcube_struct& bs)
{
	std::cout << "check bcube inite status, not yet finished" << std::endl;
	return server_establisted&&client_establisted;
}

static void sig_handler(int sig)
{
	printf("exit.\n");
	auto& bgs = bcube_gs;
	auto& bs = bgs.bcube_s;
	close(bs.server_fd);
	sleep(0.5);
	exit(0);
	return;
	for (auto& ii : bs.recv_fd)
		close(ii);
	for (auto& ii : bs.neighbor_info)
		for (auto& jj : ii)
			close(jj.remote_fd);

}
/*default is BCube(3,2)*/
void bcube_init(bcube_struct& bcube_s, bcube_global_struct& bgs)
{
	signal(SIGINT, sig_handler);
	bcube_s.bcube0_size = 3;
	bcube_s.bcube_level = 2;

	topology_init(bcube_s);
	setup_node(bcube_s);
	server_init(bcube_s);
	client_init(bcube_s);
	get_send_strategy(bcube_s);
	while (!check_bcube_is_inited_done(bcube_s))
		std::this_thread::sleep_for(std::chrono::seconds(1));
	return;
}

extern bcube_global_struct bcube_gs;


void show_msg(void* row_data)
{
	return;
	msg_struct* msg = (msg_struct*)row_data;
	printf("msg info:\n");
	printf("msg_length: %d\n",msg->msg_length);
	printf("name_length: %d\n",msg->name_len);
	printf("start position: %d\n",msg->start_pos);
	printf("msg.data[0]: %c\n",msg->data[0]);
	char* name = (char*)msg + sizeof(msg_struct);
	char* data = name +msg->name_len;
	char tmp = *data;
	*data = 0;
	printf("msg_name: %s\n", name);
	*data = tmp;
	if(0)
	{
		for (int ii = 0; ii < 3; ii++)
			printf("%d ",((int*)data)[ii]);
	}
	printf("\n");
}
extern void show_msg(void*);
static void send_assist_thread(tensor_table_entry& a_tensor, process& ps, int pid)
{
	msg_struct* tmp_msg = nullptr;
	auto& tmp_stg = ps[pid];
	for (auto it : tmp_stg)
	{
		int len = 0;
		tensor_msg::encode(a_tensor, (void**)&tmp_msg, it.paraid[0], it.block_num, &len);
		//printf("send out: %s,\t send len=%d\n",a_tensor.tensor_name.c_str(),len);
		show_msg((void*)tmp_msg);
		//assert(write(it.socket_fd, (void*)(tmp_msg), len) == len);
		//assert(send(it.socket_fd, (void*)(tmp_msg), len, 0) == len);
		size_t numsss=send(it.socket_fd, (void*)(tmp_msg), len, 0);
		if(numsss!=len)
		{
			printf("send error .........................\n");
			exit(0);
		}
		//printf("send out %d: %s,\t send len=%d\n",it.socket_fd,a_tensor.tensor_name.c_str(),len);
		std::free(tmp_msg);
		//printf("in send_assist_thread : free %p\n", tmp_msg);
		tmp_msg = nullptr;
	}
}
#include <condition_variable>
struct thread_assis
{
	std::condition_variable cv;
	std::mutex send_mutex;
	tensor_table_entry e;
	process steps;
	bool ready=false;
	std::vector<bool> fin;
} send_pack;

static bool first_init = true;

void send_thread(thread_assis& _send_pack, int pid)
{
	while (true)
	{
		printf("run in send thread\n");
		std::unique_lock<std::mutex> send_lock(_send_pack.send_mutex);
		while ((!_send_pack.ready)||_send_pack.fin[pid])_send_pack.cv.wait(send_lock);
		send_assist_thread(_send_pack.e,_send_pack.steps,pid);
		_send_pack.fin[pid] = true;
	}
	
}
#include <pthread.h> 
void* send_thread_2(void* _pid)
{
	auto pid = *(int*)_pid;
	thread_assis& _send_pack = send_pack;
	while (true)
	{
		std::unique_lock<std::mutex> send_lock(_send_pack.send_mutex);
		while ((!_send_pack.ready) || _send_pack.fin[pid])_send_pack.cv.wait(send_lock);
		send_assist_thread(_send_pack.e, _send_pack.steps, pid);
		_send_pack.fin[pid] = true;
	}
	return NULL;
}

void bcube_send2222(tensor_table_entry& e, bcube_struct& bs, int stage)
{
	auto& send_strategy = bs.my_strategy;
	assert((size_t)stage < send_strategy.size());
	auto & steps = send_strategy[stage];
	auto& _send_pack_here = send_pack;

	//std::cout<<"current send is belong to stage "<<stage<<std::endl;
	if (first_init)
	{
		first_init = false;
		_send_pack_here.fin.resize(2);
		for (int nums = 0; nums < 2; nums++)
		{
			pthread_t id1;
			if (pthread_create(&id1,NULL,send_thread_2,(void*)&nums) != 0)
			{
				printf("error in create thread");
				while (1);
			}
			sleep(2);
		}
	}
	std::unique_lock<std::mutex> send_lock(_send_pack_here.send_mutex);
	_send_pack_here.steps = steps;
	_send_pack_here.e = e;
	_send_pack_here.fin[0] = false;
	_send_pack_here.fin[1] = false;
	_send_pack_here.ready = true;
	_send_pack_here.cv.notify_all();
	send_lock.unlock();

	while (_send_pack_here.fin[0] == false || _send_pack_here.fin[1] == false);
	return;
}
//void bcube_send(tensor_table_entry& e, bcube_struct& bs, int stage)
//{
//	auto& send_strategy = bs.my_strategy;
//	assert((size_t)stage < send_strategy.size());
//	auto & steps = send_strategy[stage];
//	auto& _send_pack_here = send_pack;
//
//	//std::cout<<"current send is belong to stage "<<stage<<std::endl;
//	if (first_init)
//	{
//		first_init = false;
//		_send_pack_here.fin.resize(2);
//		for (int nums = 0; nums < 2; nums++)
//		{
//
//			std::thread(send_thread, std::ref(_send_pack_here), nums);
//			printf("create a thread...\n");
//			std::this_thread::sleep_for(std::chrono::seconds(2));
//
//		}
//	}
//	std::unique_lock<std::mutex> send_lock(_send_pack_here.send_mutex);
//	_send_pack_here.steps = steps;
//	_send_pack_here.e = e;
//	_send_pack_here.fin[0] = false;
//	_send_pack_here.fin[1] = false;
//	_send_pack_here.ready = true;
//	_send_pack_here.cv.notify_all();
//
//	while (_send_pack_here.fin[0] == false || _send_pack_here.fin[1] == false);
//	//printf("%s in stage %d is send out...\n",e.tensor_name.c_str(), stage);
//	//while (stage == 1);
//	/*send out...*/
//	return;
//}
void bcube_send(tensor_table_entry& e, bcube_struct& bs, int stage)
{
	auto& send_strategy = bs.my_strategy;
	assert((size_t)stage < send_strategy.size());
	auto & steps = send_strategy[stage];

	//std::cout<<"current send is belong to stage "<<stage<<std::endl;

	std::thread p0 = std::thread(send_assist_thread, std::ref(e), std::ref(steps), 0);
	std::thread p1 = std::thread(send_assist_thread, std::ref(e), std::ref(steps), 1);
	p0.join();
	p1.join();
	printf("%s in stage %d is send out...\n",e.tensor_name.c_str(), stage);
	//while (stage == 1);
	/*send out...*/
	return;
}



void bcube_test(void)
{
	bcube_init(bcube_gs.bcube_s, bcube_gs);
	std::cout << "bcube init done" << std::endl;
	while (1);
}
