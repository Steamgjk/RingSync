#ifndef __TENSORFLOW_RING_RDMA_H__
#define __TENSORFLOW_RING_RDMA_H__
#include <vector>
#include <string>
#include <pthread.h>
#include <stdarg.h>
#include <sys/time.h>
#include <unistd.h>
#if HAVE_RDMA
#include <rdma/rdma_cma.h>

#define MAX_CONCURRENCY 20

void log_info(const char *format, ...);

typedef void (*pre_conn_cb_fn)(struct rdma_cm_id *id);
typedef void (*connect_cb_fn)(struct rdma_cm_id *id);
typedef void (*completion_cb_fn)(struct ibv_wc *wc);
typedef void (*disconnect_cb_fn)(struct rdma_cm_id *id);

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

typedef struct _ack
{
	int index;

} _ack_;

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

typedef struct _key_exchange_
{
	int id;
	uint64_t md5;
	struct
	{
		uint64_t addr;
		uint32_t rkey;
	} key_info[MAX_CONCURRENCY];

} _key_exch;

typedef struct _e_
{
//register buffer for remote to write
	char *           buffer[MAX_CONCURRENCY];
	struct ibv_mr *  buffer_mr[MAX_CONCURRENCY];

//register ack mem is used for write to remote
	_ack_*			 ack[MAX_CONCURRENCY];
	struct ibv_mr *  ack_mr[MAX_CONCURRENCY];

	uint64_t peer_addr[MAX_CONCURRENCY];
	uint32_t peer_rkey[MAX_CONCURRENCY];

//indicate current status of each peer exchange
	int 			 is_busy[MAX_CONCURRENCY];
// index 0: store for local as tx    (to send)
// index 1: used to recv the remote info (to recv)
	_key_exch*       k_exch[2];
	struct ibv_mr*   k_exch_mr[2];
} extend_info;


//for server to use
struct conn_context
{
	char *buffer;
	struct ibv_mr *buffer_mr;

	struct message *msg;
	struct ibv_mr *msg_mr;

	FILE* fd;
	//char file_name[MAX_FILE_NAME];

	extend_info s_new_ctx;
};

//for client to use
struct client_context
{
	char *buffer;
	struct ibv_mr *buffer_mr;

	struct message *msg;
	struct ibv_mr *msg_mr;

	uint64_t peer_addr;
	uint32_t peer_rkey;

	FILE* fd;
	const char *file_name;

	// for client used
	extend_info c_new_ctx;

};




void test_rdma_header();

//Client
struct rdma_cm_id* rdma_client_init_connection(char* local_ip, char* remote_ip, int remote_port);
void* client_polling_send(struct rdma_cm_id *id);
void rdma_send_data(struct ibv_wc *wc, void* data2send, size_t data_len);
void send_tensor(struct rdma_cm_id *id, char* buff, uint32_t len);
void post_receive_client(struct rdma_cm_id *id);



//Server
struct rdma_event_channel* rdma_server_init(int local_port);
struct rdma_cm_id* server_wait4conn(struct rdma_event_channel *event_channel);
void *polling_recv_cq(struct rdma_cm_id *id); // thread  to change
int recv4data(struct ibv_wc *wc, void*& data_ptr);
void* recv_by_RDMA(struct ibv_wc *wc, uint32_t& recv_len);


struct ibv_pd * rc_get_pd();


void rc_die(const char *reason);
void printWCode(struct ibv_wc *wc);

#define TEST_NZ(x) do { if ( (x)) rc_die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) rc_die("error: " #x " failed (returned zero/null)."); } while (0)

#endif
#endif