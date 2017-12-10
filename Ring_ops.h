#ifndef __TENSOTFLOW_BCUBE_OPS__
#define __TENSOTFLOW_BCUBE_OPS__
#include "Ring_ops.h"
#include <vector>
#include <iostream>
#include <atomic>
#include <thread>
#include <string>
#include <mutex>
#include "bcube_comm.h"
#include "bcube_message.h"
#include <unordered_map>
#include <queue>

typedef char* tensor;
/*
*全局结构，包含全部的
*/

typedef std::unordered_map<std::string, std::vector<received_tensor_entry>> Received_tensor;

struct bcube_global_struct
{
	std::atomic_flag bgthread_start = ATOMIC_FLAG_INIT;
	std::vector<std::thread> bg_thread;/*两个后台thread,主线程发送,附线程接收数据*/
	std::vector<step> all_send_strategy;/*发送策略*/

	bool is_inited_done = false; /*flag indicating inited done*/

	std::mutex bcube_mutex;/*multi thread for mutex*/
	std::mutex tensor_gene_mutex;/*muthex between tensor gen and fetch*/
	std::mutex tensor_recv_mutex;/*mutex between tensor receive and reduce*/

	std::queue<tensor_table_entry> tensor_table;/*存放新加进来的tensor_表*/
	std::vector<std::vector<tensor_table_entry>> unfinished_tensor;/*2D for different stage tensor.*/
	Received_tensor receiv_tensor;/*this is storing ready to reduce tensor*/
	Received_tensor receiv_tmp_tensor;/*first insert into tmp buf, if collected all other, copy to receiv_msg.*/

	int rank;/*my rank*/
	bcube_struct bcube_s;/*bcube structs*/




	~bcube_global_struct()
	{
		std::cout << "end in global states" << std::endl;
	}
};


void bcube_all_init_onice(bcube_global_struct&);
void bcube_ops_test(void);

#endif