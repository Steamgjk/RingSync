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

void test_rdma_header();

//Client
struct rdma_cm_id* rdma_client_init_connection(char* local_ip, char* remote_ip, int remote_port);
void* client_polling_send(struct rdma_cm_id *id);
void rdma_send_data(struct ibv_wc *wc, void* data2send, size_t data_len);



//Server
struct rdma_event_channel* rdma_server_init(int local_port);
struct rdma_cm_id* server_wait4conn(struct rdma_event_channel *event_channel);
void *polling_recv_cq(struct rdma_cm_id *id); // thread  to change
int recv4data(struct ibv_wc *wc, void* data_ptr);
void* recv_by_RDMA(struct ibv_wc *wc, uint32_t& recv_len);

void rc_die(const char *reason);
void printWCode(struct ibv_wc *wc);

#define TEST_NZ(x) do { if ( (x)) rc_die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) rc_die("error: " #x " failed (returned zero/null)."); } while (0)

#endif
#endif