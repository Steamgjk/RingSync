# ==============================================================================
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
CC = g++  
WARNING=  -Wall
CFLAGS += $(WARNING) -g  -std=c++11 -Os
LDFLAGS += -lm  -pthread
PROC_NAME += proc
HEAD_DEPEND += bcube_comm.h bcube_message.h bcube_ops.h bcube_utils.h
SOURCE +=  bcube_comm.cpp bcube_message.cpp bcube_ops.cpp bcube_test.cpp bcube_utils.cpp
OBJS_DEPEND=$(subst .cpp,.o, $(SOURCE))
all: clean $(PROC_NAME)
	@echo 'make done'

# genetate objs
%.o:%.cpp $(HEAD_DEPEND)
	$(CC) -c $(CFLAGS) $< -o $@

$(PROC_NAME):$(OBJS_DEPEND)
	$(CC) $^ -o $@ $(LDFLAGS)

run:all
	./$(PROC_NAME)

.PHONY:clean
clean:  
	rm -rf $(OBJS_DEPEND) 
	rm -rf ${PROC_NAME}

