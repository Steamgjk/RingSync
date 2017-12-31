#ifndef __TENSORFLOW_RDMA_H__
#define __TENSORFLOW_RDMA_H__

#define HVAE_RDMA 1
#if HVAE_RDMA

#include <vector>
#include <string>
#include <rdma/rdma_cma.h>
#include <thread>
#include <iostream>
#include <unistd.h>


void rc_die(const char *reason);

const size_t BUFFER_SIZE = 512 * 1024 * 1024 + 1;
#define TIMEOUT_IN_MS 500
#define TEST_NZ(x) do { if ( (x)) rc_die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) rc_die("error: " #x " failed (returned zero/null)."); } while (0)
#define MIN_CQE 10

enum message_id
{
	MSG_INVALID = 0,
	MSG_MR,
	MSG_READY,
	MSG_DONE
};
struct message
{
	int id;
	union
	{
		struct
		{
			uint64_t addr;
			uint32_t rkey;
		} mr;
	} data;
};

struct context
{
	char *buffer;
	struct ibv_context *ibv_ctx;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_comp_channel *comp_channel;
	struct ibv_mr *buffer_mr;
	struct message *msg;
	struct ibv_mr *msg_mr;
	std::thread  cq_poller_thread;
	uint64_t peer_addr;
	uint32_t peer_rkey;
	bool remote_idle;
};


struct _recv_chain
{
	void* data_ptr;
	int data_len;
	_recv_chain* next;
};
#endif // HAVE_RDMA
#endif // __TENSORFLOW_RDMA_H__
