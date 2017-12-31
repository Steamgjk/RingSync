#ifndef __TENSORFLOW_RING_RDMA_H__
#define __TENSORFLOW_RING_RDMA_H__
#include <vector>
#include <string>
#include <pthread.h>
#if HAVE_RDMA
#include <rdma/rdma_cma.h>


typedef struct _data_list_
{
	char* data_ptr;
	struct _data_list_* next;
} node_item;

typedef struct _rdma_pack_
{
	struct rdma_cm_id* rdma_id;
	node_item* nit;
} _rdma_thread_pack_;

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
	pthread_t cq_poller_thread;
	uint64_t peer_addr;
	uint32_t peer_rkey;
	bool remote_idle;
};

void rdma_bcube_init(bcube_struct&, bcube_global_struct&);
void rdma_bcube_send(tensor_table_entry& , bcube_struct& , int );
#endif
#endif