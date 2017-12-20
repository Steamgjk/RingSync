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
#include "bcube_message.h"
#include <assert.h>
#include <cstring>

/*
e need to be allocated before
*/
//int TYPE_SIZE[] = { 1, 4, 8, 4, 8 };
static  int TYPE_SIZE[] = 
{ 
	4,				sizeof(bool), 
	sizeof(uint8_t), sizeof(int8_t), 
	sizeof(uint16_t),sizeof(int16_t),
	sizeof(uint32_t),sizeof(int32_t),
	sizeof(uint64_t),sizeof(int64_t),
	sizeof(float_t),sizeof(double_t)
};


#include "bcube_comm.h"
static void show_msg(void* row_data)
{
	return;
	msg_struct* msg = (msg_struct*)row_data;
	printf("msg info:\n");
	printf("msg_length: %d\n", msg->msg_length);
	printf("name_length: %d\n", msg->name_len);
	printf("start position: %d\n", msg->start_pos);
	printf("msg.data[0]: %c\n", msg->data[0]);
	char* name = (char*)msg + sizeof(msg_struct);
	char* data = name + msg->name_len;
	char tmp = *data;
	*data = 0;
	printf("msg_name: %s\n", name);
	*data = tmp;
	if(0)
	{
		for (int ii = 0; ii < 3; ii++)
			printf("%d ", ((int*)data)[ii]);
	}
	printf("\n");
}

void tensor_msg::decode(received_tensor_entry& e, void* msg)
{
	assert(msg != nullptr);
	msg_struct* msg_tp = (msg_struct*)msg;
	assert(msg_tp->data[0] == ',');
	show_msg(msg);
	char* tensor_name = (char*)((char*)msg + sizeof(msg_struct));
	char* tensor_data = (char*)(tensor_name + msg_tp->name_len);
	if (msg_tp->t_ops == ALLREDUCE)
	{
		char tmp_char = *tensor_data;
		*tensor_data = 0;
		e.tensor_name = std::string(tensor_name);
		*tensor_data = tmp_char;
		e.start_position = msg_tp->start_pos;
		e.tensor_nums = msg_tp->nums;
		int tensor_length = (msg_tp->nums)*TYPE_SIZE[(msg_tp->tensor_type)];
		e.receive_ptr = (char*)std::malloc(tensor_length);
		assert(e.receive_ptr != nullptr);
		memset(e.receive_ptr, 0, tensor_length);
		std::memcpy(e.receive_ptr, tensor_data, tensor_length);
		return;
	}else if(msg_tp->t_ops == ALLGATHER|| msg_tp->t_ops == BROADCAST)
	{
		int type_size = TYPE_SIZE[msg_tp->tensor_type];
		char* tensor_shape = (char*)(tensor_data + (msg_tp->nums)*type_size);
		char* stop_pos = (char*)msg + msg_tp->msg_length;

		char tmp_char = *tensor_data;
		*tensor_data = 0;
		e.tensor_name = std::string(tensor_name);
		*tensor_data = tmp_char;

		e.start_position = msg_tp->start_pos;
		e.tensor_nums = msg_tp->nums;
		while (tensor_shape < stop_pos)
		{
			Tensor_Info _part_tensor;
			auto block_size = *(int*)tensor_shape;
			_part_tensor.tensor_shape = block_size;
			_part_tensor.tensor_ptr = (void*)std::malloc(block_size*type_size);
			assert(_part_tensor.tensor_ptr != nullptr);
			std::memcpy(_part_tensor.tensor_ptr,
				tensor_data, block_size*type_size);
			tensor_data += block_size*type_size;
			e.gather_ptr.push_back(std::move(_part_tensor));
			tensor_shape += sizeof(int);
		}
		return;
	}
	else
	{
		perror("error in tensor ops check...\n");
		exit(-1);
	}
	return;
}


/*
encode e to msg. malloc msg memory here.
*/
void tensor_msg::encode(tensor_table_entry& e, void** msg, 
	int start_pos, int block_nums, int* total_length)
{
	if (e.tensor_ops == ALLREDUCE)
	{
		int element_nums = e.block_size*block_nums;
		auto type_size = TYPE_SIZE[e.tensor_type];
		auto tensor_size = type_size*element_nums;
		*total_length = sizeof(msg_struct) + e.tensor_name.length() + tensor_size;
		auto malloc_ptr = (char*)std::malloc(*total_length);
		assert(malloc_ptr != nullptr);
		memset(malloc_ptr, 0, *total_length);
		auto msg_ptr = (msg_struct*)malloc_ptr;
		msg_ptr->name_len = e.tensor_name.length();
		msg_ptr->data[0] = ',';/*flags*/
		msg_ptr->nums = element_nums;
		msg_ptr->tensor_type = e.tensor_type;
		msg_ptr->start_pos = start_pos;
		msg_ptr->msg_length = *total_length;
		msg_ptr->t_ops = e.tensor_ops;

		char* name_position = (char*)((char*)malloc_ptr + sizeof(msg_struct));
		char* tensor_position = (char*)(name_position + msg_ptr->name_len);
		std::memcpy(name_position, e.tensor_name.c_str(), msg_ptr->name_len);
		std::memcpy(tensor_position, (void*)((char*)e.tensor_data + start_pos*e.block_size*type_size), tensor_size);
		*msg = malloc_ptr;
		return;
	}
	else if (e.tensor_ops == ALLGATHER|| e.tensor_ops == BROADCAST)
	{
/*
|-------------------------------------------------------------------------------------------------------------
|index(byte):|[0,3]|  [4,7]  |   [8,11]  |[12,15]|[16,19] |  [20,23]  |  [24,27] |28|[29,31]|[32,31+name_len]|
|------------|-----|---------|-----------|-------|--------|-----------|----------|--|-------|----------------|
|data(name) :|rank |start_pos|tensor_nums|msg_len|name_len|tensor_type|tensor_ops|, | unuse |   tensor name  |
|-------------------------------------------------------------------------------------------------------------
|index(byte):|[32+name_len,31+name_len+tensor_nums*type_size]|[32+name_len+tensor_nums*type_size,msg_len-1]  |
|------------|-----------------------------------------------|-----------------------------------------------|
|data(name) :|             tensor data                       |              tensor shape                     |
|-------------------------------------------------------------------------------------------------------------
*/
		/*allgather or broadcast*/
		int element_nums = 0;
		for (int block_index = 0; block_index < block_nums; block_index++)
			element_nums += e.gather_tensor[start_pos + block_index].tensor_shape;

		auto type_size = TYPE_SIZE[e.tensor_type];
		auto tensor_size = type_size*element_nums;
		*total_length = sizeof(msg_struct) + e.tensor_name.length() + tensor_size + block_nums*sizeof(int);
		auto malloc_ptr = (char*)std::malloc(*total_length);
		assert(malloc_ptr != nullptr);
		memset(malloc_ptr, 0, *total_length);
		auto msg_ptr = (msg_struct*)malloc_ptr;
		msg_ptr->name_len = e.tensor_name.length();
		msg_ptr->data[0] = ',';/*flags*/
		msg_ptr->nums = element_nums;
		msg_ptr->tensor_type = e.tensor_type;
		msg_ptr->start_pos = start_pos;
		msg_ptr->msg_length = *total_length;
		msg_ptr->t_ops = e.tensor_ops;

		char* name_position = (char*)((char*)malloc_ptr + sizeof(msg_struct));
		char* tensor_position = (char*)(name_position + msg_ptr->name_len);
		int* shape_position = (int*)(tensor_position + tensor_size);

		std::memcpy(name_position, e.tensor_name.c_str(), msg_ptr->name_len);
		for (int block_index = 0; block_index < block_nums; block_index++)
		{
			*(shape_position) = e.gather_tensor[start_pos + block_index].tensor_shape;
			assert(tensor_position<(malloc_ptr+msg_ptr->msg_length));
			std::memcpy(tensor_position, e.gather_tensor[start_pos + block_index].tensor_ptr, 
				(*shape_position)*type_size);
			tensor_position += (*shape_position)*type_size;			
			shape_position++;
		}
		*msg = malloc_ptr;
	}
	else
	{
		perror("error in unkonwn tensor ops\n");
	}
}

