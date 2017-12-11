CC = g++  
CFLAGS = -Wall -g -std=c++11  
LDFLAGS = -lm -pthread 
PROC_NAME = gjk_proc
CSOURCE=Main.cpp MyRing.cpp 
HEAD=MyRing.h
OBJS=$(subst .cpp,.o,$(CSOURCE))

all: clean MyRing.o ${PROC_NAME}

%.o : %.cpp  $(HEAD)
	${CC} ${CFLAGS} -c $< -o $@

${PROC_NAME} : $(OBJS)
	${CC} ${LDFLAGS} $^  -o ${PROC_NAME}   
clean:  
	rm -rf $(OBJS)  
	rm -rf ${PROC_NAME}  