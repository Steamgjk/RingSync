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

#include <cstdio>
#include <iostream>
#include <vector>
#include "bcube_comm.h"
#include "bcube_ops.h"
/*
*socket asy
* http://blog.csdn.net/bian1029/article/details/72974505
*/

void allreduce_test(bcube_global_struct& bcube_gs)
{
	int loops = 0;
	while (1)
	{
		std::vector<std::thread> thread_handle;
		int threadnum = 60;
		while (threadnum--)
		{
			thread_handle.push_back(std::thread(allreduce_enque, std::ref(bcube_gs), threadnum, loops));
		}
		for (auto& thread_id : thread_handle)
			thread_id.join();
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		loops++;
	}
}
void allgather_test(bcube_global_struct& bcube_gs)
{
	int loops = 0;
	while (1)
	{
		std::vector<std::thread> thread_handle;
		int threadnum = 12;
		while (threadnum--)
		{
			thread_handle.push_back(std::thread(allgather_enqueue, std::ref(bcube_gs), threadnum, loops));
		}
		for (auto& thread_id : thread_handle)
			thread_id.join();
		//std::cout << "create tensors done" << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		int cycle = 0;
		int receive_size = 0;
		while (loops > (reduce_loop + 20))
		{
			cycle++;
			int rrr = (int)bcube_gs.receiv_tmp_tensor.size();
			if (receive_size != rrr)
			{
				receive_size = rrr;
				cycle = 0;
			}
			//printf("in %d loops, receiv_tmp_tensor.size() = %d\n", loops, rrr);
			std::this_thread::sleep_for(std::chrono::seconds(1));
			if ((cycle > 10) && rrr == receive_size)
				exit(-1);
		}
		//std::this_thread::sleep_for(std::chrono::seconds(10));
		loops++;
	}
}
void broadcast_test(bcube_global_struct& bcube_gs)
{
	int loops = 0;
	while (1)
	{
		std::vector<std::thread> thread_handle;
		int threadnum = 12;
		while (threadnum--)
		{
			thread_handle.push_back(std::thread(broadcast_enqueue, std::ref(bcube_gs), threadnum, loops));
		}
		for (auto& thread_id : thread_handle)
			thread_id.join();
		//std::cout << "create tensors done" << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		//std::this_thread::sleep_for(std::chrono::seconds(10));
		loops++;
	}
}


void bcube_ops_test(void)
{
	bcube_all_init_onice(bcube_gs);

	allreduce_test(bcube_gs);
	//allgather_test(bcube_gs);
	broadcast_test(bcube_gs);

	return;
}

int main(int argc, char** argv)
{

	bcube_ops_test();
	return 0;
}
