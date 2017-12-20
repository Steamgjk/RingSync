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

#ifndef UTILS_H
#define UTILS_H
#include <cstdio>

#define MAX_N 1000
using namespace std;



class Utils
{
public:
	Utils();
	static int Convert(int temp[], int len, int k);
	static void Convert2Array(int val, int k, int*ans);
	static void getOneHopNeighbour(int serv_id, int l, int n, int k, int*one_hop_neighours);
	static void getScatterMatrix(int p, int s, int n, int k, int N, int**sendMatrix, int&para_num);
	static void getGatherMatrix(int p, int s, int n, int k, int N, int**sendMatrix, int&para_num);
	~Utils();
private:

};
#endif
