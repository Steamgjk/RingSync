#include "Ring_ops.h"
#include <iostream>
#include <atomic>
#include <thread>
#include <string>
#include <assert.h>
#include <cstring>



bcube_global_struct bcube_gs;
void bcube_do_steps(bcube_global_struct&);
void bg_loops(bcube_global_struct& bgs)
{

	bcube_init(bgs.bcube_s, bgs);
	bgs.unfinished_tensor.resize(4);
	bgs.is_inited_done = true;
	std::cout << "all init done, now we are going to send msg in bgthread..." << std::endl;
	while (true)
	{
		//std::cout << "run in bg_main_threads, now we are going to send some message" << std::endl;
		try
		{
			bcube_do_steps(bgs);
		}
		catch (exception& e)
		{
			std::cout << "error: " << e.what();
			while (1);
		}
		//std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}


/*
全局init，开启一个后台线程.
*/

void bcube_all_init_onice(bcube_global_struct& gs)
{
	static int print_count = 0;
	if (!bcube_gs.bgthread_start.test_and_set())
	{
		std::cout << "start the bgthread" << std::endl;
		bcube_gs.bg_thread.push_back(std::thread(bg_loops, std::ref(bcube_gs)));
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	while (!bcube_gs.is_inited_done)
	{
		if ((print_count++) % 5 == 0)
			std::cout << "bcube is not inited successfully, waiting for other connecting" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}



bool bcube_reduce(bcube_global_struct& bgs, tensor_table_entry& e, bool is_scatter)
{
	auto tensor_name = e.tensor_name;
	std::vector<received_tensor_entry> rcv_tensor;
	{
		std::lock_guard<std::mutex> rece_lock(bgs.bcube_mutex);
		auto& tensor_receive = bgs.receiv_tensor;
		auto find_tensor = tensor_receive.find(tensor_name);
		if (find_tensor == tensor_receive.end())return false;/*没有收到,下一次再取*/
		rcv_tensor = std::move(find_tensor->second);
		tensor_receive.erase(find_tensor);
	}
	assert(rcv_tensor.size() == 4);
	//return true;
	if (e.tensor_ops == ALLREDUCE)
	{
		if (is_scatter)/*for scatter, sum these tensor*/
		{

			/*下面做一次加和操作*/
			for (auto it = rcv_tensor.begin(); it != rcv_tensor.end(); it++)
			{
				auto tensor_ptr = (int*)it->receive_ptr;
				auto tensor_counts = it->tensor_nums;
				auto start_position = it->start_position;
				auto e_tensor_ptr = e.tensor_data;
				auto type_size = TYPE_SIZE[e.tensor_type];
				auto block_size = e.block_size;
				auto add_pos = (int*)((char*)e_tensor_ptr + start_position * type_size * block_size);
				for (size_t addnum = 0; addnum < tensor_counts; addnum++)
				{
					//printf("%d,%d,%d\n", add_pos[addnum], tensor_ptr[addnum], add_pos[addnum]+ tensor_ptr[addnum]);
					add_pos[addnum] += tensor_ptr[addnum];

				}
				{
					/*release reources*/
					std::free(it->receive_ptr);
					//printf("in allreduce: free %p\n", it->receive_ptr);
					it->receive_ptr = nullptr;
				}
			}
			return true;
		}
		else/*gather, replace these tensor*/
		{
			/*下面做一次替换操作*/
			for (auto it = rcv_tensor.begin(); it != rcv_tensor.end(); it++)
			{
				auto tensor_ptr = (int*)it->receive_ptr;
				auto tensor_counts = it->tensor_nums;
				auto start_position = it->start_position;
				auto e_tensor_ptr = e.tensor_data;
				auto type_size = TYPE_SIZE[e.tensor_type];
				auto block_size = e.block_size;
				auto add_pos = (int*)((char*)e_tensor_ptr + start_position * type_size * block_size);
				for (size_t addnum = 0; addnum < tensor_counts; addnum++)
				{
					//printf("%d,%d,%d\n", add_pos[addnum], tensor_ptr[addnum], tensor_ptr[addnum]);
					add_pos[addnum] = tensor_ptr[addnum];
				}
				{
					/*release reources*/
					std::free(it->receive_ptr);
					//printf("in allreduce: free %p\n", it->receive_ptr);
					it->receive_ptr = nullptr;
				}
			}
			return true;
		}
	}
	else if (e.tensor_ops == ALLGATHER || e.tensor_ops == BROADCAST)
	{
		for (auto it = rcv_tensor.begin(); it != rcv_tensor.end(); it++)
		{
			for (size_t index = 0; index < it->gather_ptr.size(); index++)
			{
				auto& _a_tensor = e.gather_tensor[it->start_position + index];
				/*first to do:
					release the old resource
					std::free(_a_tensor.tensor_ptr);
					_a_tensor.tensor_ptr=nullptr;
					,but now, we will release them last.
				*/
				std::free(_a_tensor.tensor_ptr);
				//printf("in allgather reduce: free %p\n", _a_tensor.tensor_ptr);
				_a_tensor.tensor_ptr = it->gather_ptr[index].tensor_ptr;
				_a_tensor.tensor_shape = it->gather_ptr[index].tensor_shape;
			}
		}
	}
	return true;
}



static int reduce_loop = 0;


void allreduce_enque(bcube_global_struct& bgs, int unique_id, int loops)
{
	std::random_device rd;
	int init_num = rd() % 83;
	tensor_table_entry e;
	e.tensor_name = "_" + std::to_string(loops) + "_allreduce_" + std::to_string(unique_id);
	e.tensor_type = T_INT32;
	e.tensor_ops = ALLREDUCE;

	bool padding = true;
	e.available_nums = loops / 100 + 1;

	int* a_ = new int[e.available_nums]();
	for (int nums = 0; nums < e.available_nums; nums++)a_[nums] = init_num++;

	if (padding)
	{
		/*padding...*/
		e.block_size = (e.available_nums + 17) / 18;
		e.tensor_size = 18 * e.block_size;
		e.tensor_data = std::malloc(e.tensor_size * sizeof(int));
		assert(e.tensor_data != nullptr);
		std::memcpy(e.tensor_data, (void*)a_, e.available_nums * sizeof(int));
		std::free(a_);
	}
	else
	{
		e.block_size = 1;
		e.available_nums = 18;
		e.tensor_size = 18;
		e.tensor_data = (void*)a_;
	}

#ifdef _DEBUG_TENSOR_GEN_show_
	{
		std::lock_guard<std::mutex> print_tensor(bgs.bcube_mutex);
		printf("creates %s : [", e.tensor_name.c_str());
		for (int i = 0; i < 18; i++)printf("%5d", a_[i]);
		printf(" ]\n");
	}
#endif
	{
		std::lock_guard<std::mutex> enque_lock(bgs.bcube_mutex);
		auto& tensor_table = bgs.tensor_table;
		tensor_table.push(std::move(e));
	}
	while (bgs.tensor_table.size() > 100)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	return;
}




void bcube_ops_test(void)
{
	bcube_all_init_onice(bcube_gs);

	allreduce_test(bcube_gs);


	return;
}