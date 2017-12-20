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

#include "bcube_utils.h"

int Utils::Convert(int temp[], int len, int k)
{
	int sum = 0;
	int i = len - 1;
	for (i = len - 1; i >= 0; i--)
	{
		sum = sum * k + temp[i];
	}
	return sum;
}
void Utils::Convert2Array(int val, int k, int* ans)
{
	int cnt = 0;
	while (val != 0)
	{
		ans[cnt++] = val % k;
		val = val / k;
	}
	return;

}

void Utils::getOneHopNeighbour(int serv_id, int l, int n, int k, int*one_hop_neighours)
{

	int * temp_val = new int[k + 1];
	for (int index = 0; index < k + 1; index++)temp_val[index] = 0;

	//printf("l=%d\n",l );
	Utils::Convert2Array(serv_id, n, temp_val);
	/*
	for(i = 0;i < k+1; i++)printf("%d\t", temp_val[i]);
	printf("\n");
	**/
	for (int i = 0; i < n - 1; i++)
	{
		temp_val[l] = (temp_val[l] + 1) % (n);
		int server_idx = Utils::Convert(temp_val, k + 1, n);
		one_hop_neighours[i] = server_idx;
	}
	delete[] temp_val;
}

void Utils::getScatterMatrix(int p, int s, int n, int k, int N, int**sendMatrix, int&para_num)
{
	int l = (p + s) % (k + 1);
	para_num = N;
	int * send_neighbours = new int[n - 1];
	for (int index = 0; index < n - 1; index++)send_neighbours[index] = 0;
	//int temp_val[k + 1]={0};
	//int send_neighbours[n - 1];
	for (int i = 0; i <= s; i++)
	{
		para_num /= n;
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			sendMatrix[i][j] = -1;
		}
	}
	int recv_id = 0;
	for (recv_id = 0; recv_id < N; recv_id++)
	{
		int recv_addr[100];
		for (int i = 0; i < 100; i++ )
		{
			recv_addr[i] = 0;
		}
		Utils::Convert2Array(recv_id, n, recv_addr);
		int loc = recv_addr[l];
		//int temp_val[k + 2];
		int * temp_val = new int[k +2];
		for (int index = 0; index < k+2; index++)temp_val[index] = 0;

		for (int i = 0; i <= k + 1; i++)
		{
			temp_val[i] = 0;
		}
		temp_val[k + 1] = p;
		for (int i = 0 ; i < s; i++)
		{
			temp_val[k - i] =  recv_addr[(i + p) % (k + 1)];
		}
		temp_val[k - s] = loc;
		int sta_idx = Utils::Convert(temp_val, k + 2, n);
		//printf("p:%d\ts:%d\tsta:%d\tpara_num:%d\t%d-<\t",p,s,sta_idx, para_num, recv_id );
		Utils::getOneHopNeighbour(recv_id, l, n, k, send_neighbours);
		/*
		if(recv_id == 3||recv_id == 6){
			for(i = 0; i< n-1; i++){
				printf("%d  neighbour %d\n",recv_id,send_neighbours[i] );
			}
		}
		**/
		for (int i = 0; i < n - 1; i++)
		{
			/*
			if(send_neighbours[i] == 0){
				printf("%d %d  %d\n", send_neighbours[i],recv_id, sta_idx);
			}
			**/
			sendMatrix[send_neighbours[i]][recv_id] = sta_idx;
		}
		delete[] temp_val;
	}
	delete[] send_neighbours;

}
void Utils::getGatherMatrix(int p, int s, int n, int k, int N, int**sendMatrix, int&para_num)
{
	int l =  ( p + k - s ) % (k + 1);
	para_num = N;
	int i = 0;
	int j = 0;
	for ( i = k; i >= s; i--)
	{
		para_num /= (n);
	}
	int * recv_neighbours = new int[n - 1];
	for (int index = 0; index < n - 1; index++)recv_neighbours[index] = 0;
	for ( i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			sendMatrix[i][j] = -1;
		}
	}
	int send_id = 0;
	for (send_id = 0; send_id < N; send_id++)
	{
		int send_addr[100];
		for (i = 0; i < 100; i++ )
		{
			send_addr[i] = 0;
		}
		Utils::Convert2Array(send_id, n, send_addr);

		//int temp_val[k + 2];
		int * temp_val = new int[n + 2];
		for (int index = 0; index < n + 2; index++)temp_val[index] = 0;

		for (i = 0; i <= k + 1; i++)
		{
			temp_val[i] = 0;
		}
		temp_val[k + 1] = p;
		for ( i = k; i >= s; i--)
		{
			temp_val[i] = send_addr[ ( (k) -  i + p ) % (k + 1)];
		}
		int sta_idx = Utils::Convert(temp_val, k + 2, n);

		Utils::getOneHopNeighbour(send_id, l, n, k, recv_neighbours);
		for (i = 0; i < n - 1; i++)
		{
		    //printf("send id: %d\t rev neigh: %d\t stax: %d\n",send_id,recv_neighbours[i],sta_idx);
			sendMatrix[send_id][recv_neighbours[i]] = sta_idx;
		}
		delete[]temp_val;
	}
	delete[] recv_neighbours;
	return;
}
