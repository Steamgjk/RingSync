#include "rdma_t.h"

#if HAVE_RDMA
#include <rdma/rdma_cma.h>
#include "rdma_t.h"
//void rc_die(const char *reason);

#define TIMEOUT_IN_MS 500

#define MIN_CQE 10

#endif // RDMA_SUPPORT

#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
#include <unordered_map>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>


#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>

#define IS_CLIENT false
#define IS_SERVER true

static std::atomic_bool rdma_server_establisted(false);
static std::atomic_bool rdma_client_establisted(false);

//static struct context *s_ctx = NULL;
static pre_conn_cb_fn s_on_pre_conn_cb = NULL;
static connect_cb_fn s_on_connect_cb = NULL;
static completion_cb_fn s_on_completion_cb = NULL;
static disconnect_cb_fn s_on_disconnect_cb = NULL;

void rc_die(const char *reason)
{
	extern int errno;
	fprintf(stderr, "%s\nstrerror= %s\n", reason, strerror(errno));
	exit(-1);
}

int enable_log = 1;
void log_info(const char *format, ...)
{
	if (!enable_log)return;
	char now_time[32];
	char s[1024];
	char content[1024];
	//char *ptr = content;
	struct tm *tmnow;
	struct timeval tv;
	bzero(content, 1024);
	va_list arg;
	va_start (arg, format);
	vsprintf (s, format, arg);
	va_end (arg);

	gettimeofday(&tv, NULL);
	tmnow = localtime(&tv.tv_sec);

	sprintf(now_time, "%04d/%02d/%02d %02d:%02d:%02d:%06ld ", \
	        tmnow->tm_year + 1900, tmnow->tm_mon + 1, tmnow->tm_mday, tmnow->tm_hour, \
	        tmnow->tm_min, tmnow->tm_sec, tv.tv_usec);

	sprintf(content, "%s %s", now_time, s);
	printf("%s", content);

//save_log2_file(content);
}


#if HAVE_RDMA


static node_item* get_new_node(void)
{
	node_item* nit = (node_item*)std::malloc(sizeof(node_item));
	if (nit == nullptr)
	{
		printf("fatal error : malloc node_item error\n");
		exit(-1);
	}
	nit->next = nullptr;
	nit->data_ptr = nullptr;
	return nit;
}

static _rdma_thread_pack_* get_new_thread_pack(struct rdma_cm_id* id, node_item* nit)
{
	_rdma_thread_pack_* rtp = (_rdma_thread_pack_*)std::malloc(sizeof(_rdma_thread_pack_));
	if (rtp == nullptr)
	{
		printf("fatal error : malloc _rdma_thread_pack_ error\n");
		exit(-1);
	}
	rtp->rdma_id = id;
	rtp->nit = nit;
	return rtp;
}


/*加载所有的网络节点*/
/*
12.12.10.XXX
12.12.11.XXX
*/


/*
static void send_message(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	memset(&wr, 0, sizeof(wr));

	wr.wr_id = (uintptr_t)id;
	wr.opcode = IBV_WR_SEND;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_SIGNALED;

	sge.addr = (uintptr_t)ctx->msg;
	sge.length = sizeof(*ctx->msg);
	sge.lkey = ctx->msg_mr->lkey;

	TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
}

void send_tensor(struct rdma_cm_id *id, char* buff, uint32_t len)
{
	//printf("Sending tensor...\n");
	struct context *ctx = (struct context *)id->context;
	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;
	if (buff)
		memcpy(ctx->buffer, buff, len);
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)id;
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = htonl(len);
	wr.wr.rdma.remote_addr = ctx->peer_addr;
	wr.wr.rdma.rkey = ctx->peer_rkey;
	if (len)
	{
		wr.sg_list = &sge;
		wr.num_sge = 1;

		sge.addr = (uintptr_t)ctx->buffer;
		sge.length = len;
		sge.lkey = ctx->buffer_mr->lkey;
	}
	TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
}


void post_receive_client(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	struct ibv_recv_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)id;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	sge.addr = (uintptr_t)ctx->msg;
	sge.length = sizeof(*ctx->msg);
	sge.lkey = ctx->msg_mr->lkey;
	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));
}

static void post_receive_server(struct rdma_cm_id *id)
{
	struct ibv_recv_wr wr, *bad_wr = NULL;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)id;
	wr.sg_list = NULL;
	wr.num_sge = 0;
	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));
}
**/
static char* data_gene(int size)
{
	char* _data = (char*)malloc(size * sizeof(char) + 1);
	_data[size] = 0;
	char padding[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
	for (int index = 0; index < size; index++)
		_data[index] = padding[index % 10];
	return _data;
}

static char* __send_str = "123";
static size_t __send_len = 4;
void* recv_by_RDMA(struct ibv_wc *wc, uint32_t& recv_len)
{

}
int recv4data(struct ibv_wc *wc, void*& data_ptr)
{
	return 0;
}
void send_by_RDMA(struct ibv_wc *wc)
{

}
/*
static void send_by_RDMA(struct ibv_wc *wc)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		printf("send thread %ld will never be here!!!!!\n", pthread_self());
		exit(0);
	}
	else if (wc->opcode & IBV_WC_RECV)
	{
		if (ctx->msg->id == MSG_MR)
		{
			ctx->peer_addr = ctx->msg->data.mr.addr;
			ctx->peer_rkey = ctx->msg->data.mr.rkey;
			printf("received remote memory address and key\n");
			ctx->remote_idle = true;
#if __RDMA_SLOW__
			printf("thread %ld will send data in 10 seconds\n", pthread_self());
			std::this_thread::sleep_for(std::chrono::seconds(10));
#endif
			//__send_str = data_gene(1024 * 1024 * 100);
			send_tensor(id, __send_str, __send_len);
		}
		else if (ctx->msg->id == MSG_DONE)
		{
			printf("received DONE, disconnecting\n");
			rdma_disconnect(id);
			return;
		}
		else if (ctx->msg->id == MSG_READY)
		{
			ctx->remote_idle = true;
#if __RDMA_SLOW__
			printf("thread %ld will send data in 10 seconds\n", pthread_self());
			std::this_thread::sleep_for(std::chrono::seconds(10));
#endif
			send_tensor(id, __send_str, __send_len);
		}
		post_receive_client(id);
	}
	return;
}

void* recv_by_RDMA(struct ibv_wc *wc, uint32_t& recv_len)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;
	void* _data = nullptr;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		uint32_t size = ntohl(wc->imm_data);
		struct sockaddr_in* client_addr = (struct sockaddr_in *)rdma_get_peer_addr(id);
		static int64_t lpop = 0;

		lpop++;
		//printf("%s\n",ctx->buffer);
		//msg_struct* msg = (msg_struct*)(ctx->buffer);
		//printf("recv from node %d count: %d\n", msg->rank, ++recvcount[msg->rank]);
		_data = (void*)std::malloc(sizeof(char) * size);
		recv_len = size;

		std::memcpy(_data, ctx->buffer, size);
		printf("OOOOOK\n");
		post_receive_server(id);
		ctx->msg->id = MSG_READY;
		send_message(id);
	}
	else if (wc->opcode & IBV_WC_RECV)
	{
		printf("recv thread %ld will never be here!!!!!\n", pthread_self());
		exit(0);
	}
	return _data;
}



void *polling_recv_cq(struct rdma_cm_id *id)
{
	struct ibv_cq *cq = NULL;
	struct ibv_wc wc;


	struct context *ctx = (struct context *)id->context;
	void *ev_ctx = NULL;


	while (true)
	{
		TEST_NZ(ibv_get_cq_event(ctx->comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));

		while (ibv_poll_cq(cq, 1, &wc))
		{
			if (wc.status == IBV_WC_SUCCESS)
			{
				void* recv_data = nullptr;
				int sz = recv4data(&wc, recv_data);
				if (recv_data != nullptr)//received data, will append to recv_chain...
				{
					printf("Polling Recved Data  sz = %d\n", sz);
				}
			}
			else
			{
				printf("\nwc = %s\n", ibv_wc_status_str(wc.status));
				rc_die("poll_cq: status is not IBV_WC_SUCCESS");
			}
		}
	}
	return NULL;
}

int recv4data(struct ibv_wc *wc, void*& data_ptr)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;
	//void* _data = nullptr;
	data_ptr = nullptr;
	uint32_t size = -1;
	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		size = ntohl(wc->imm_data);
		struct sockaddr_in* client_addr = (struct sockaddr_in *)rdma_get_peer_addr(id);
		static int64_t lpop = 0;

		data_ptr = (void*)std::malloc(sizeof(char) * size);
		if (data_ptr == nullptr)
		{
			printf("fatal error in recv data malloc!!!!\n");
			exit(-1);
		}
		std::memcpy(data_ptr, ctx->buffer, size);
		//printf("recv4data:Data can be gained\n");

		post_receive_server(id);
		ctx->msg->id = MSG_READY;
		send_message(id);
		//printf("send_message back\n");
	}
	else if (wc->opcode & IBV_WC_RECV)
	{
		printf("recv thread will never be here!!!!!\n");
		exit(0);
	}
	else
	{
		//printWCode(wc);
	}
	return size;
}
**/
void printWCode(struct ibv_wc *wc)
{
	switch (wc->opcode)
	{
	case IBV_WC_RECV_RDMA_WITH_IMM:
		printf("IBV_WC_RECV_RDMA_WITH_IMM\n");
		break;
	case IBV_WC_SEND:
		printf("IBV_WC_SEND\n");
		break;
	case IBV_WC_RDMA_WRITE:
		printf("IBV_WC_RDMA_WRITE\n");
		break;
	case IBV_WC_RDMA_READ:
		printf("IBV_WC_RDMA_READ\n");
		break;
	case IBV_WC_COMP_SWAP:
		printf("IBV_WC_COMP_SWAP\n");
		break;
	case IBV_WC_FETCH_ADD:
		printf("IBV_WC_FETCH_ADD\n");
		break;
	case IBV_WC_BIND_MW:
		printf("IBV_WC_BIND_MW\n");
		break;
	case IBV_WC_RECV:
		printf("IBV_WC_RECV\n");
		break;
	default:
		printf("Unknown\n");
	}
}
void *client_polling_send(struct rdma_cm_id *id)
{
	struct ibv_cq *cq = NULL;
	struct ibv_wc wc;
	struct context *ctx = (struct context *)id->context;
	void *ev_ctx = NULL;

	while (1)
	{
		TEST_NZ(ibv_get_cq_event(ctx->comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));

		while (ibv_poll_cq(cq, 1, &wc))
		{
			if (wc.status == IBV_WC_SUCCESS)
			{
				send_by_RDMA(&wc);
			}
			else
			{
				printf("\nwc = %s\n", ibv_wc_status_str(wc.status));
				rc_die("poll_cq1: status is not IBV_WC_SUCCESS");
			}
		}
	}
	return NULL;
}

static void *send_poll_cq(void *tmp_id)
{
	struct ibv_cq *cq = NULL;
	struct ibv_wc wc;
	struct rdma_cm_id *id = (struct rdma_cm_id *)tmp_id;
	struct context *ctx = (struct context *)id->context;
	void *ev_ctx = NULL;

	while (1)
	{
		TEST_NZ(ibv_get_cq_event(ctx->comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));

		while (ibv_poll_cq(cq, 1, &wc))
		{
			if (wc.status == IBV_WC_SUCCESS)
			{
				send_by_RDMA(&wc);
			}
			else
			{
				printf("\nwc = %s\n", ibv_wc_status_str(wc.status));
				rc_die("poll_cq2: status is not IBV_WC_SUCCESS");
			}
		}
	}
	return NULL;
}
/*
void rdma_send_data(struct ibv_wc *wc, void* data2send, size_t data_len)
{
	//printf("Here:rdma_send_data....\n");
	//printWCode(wc);
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		printf("send thread %ld will never be here!!!!!\n", pthread_self());
		exit(0);
	}
	else if (wc->opcode & IBV_WC_RECV)
	{
		if (ctx->msg->id == MSG_MR)
		{
			ctx->peer_addr = ctx->msg->data.mr.addr;
			ctx->peer_rkey = ctx->msg->data.mr.rkey;
			//printf("received remote memory address and key\n");
			ctx->remote_idle = true;

			//__send_str = data_gene(1024 * 1024 * 100);
			send_tensor(id, (char*)data2send, data_len);
			//printf("INIt SEnd\n");
		}
		else if (ctx->msg->id == MSG_DONE)
		{
			//printf("received DONE, disconnecting\n");
			rdma_disconnect(id);
			return;
		}
		else if (ctx->msg->id == MSG_READY)
		{
			ctx->remote_idle = true;
			//printf("COns Send\n");
			send_tensor(id, (char*)data2send, data_len);
			//printf("Adter Send\n");
		}
		post_receive_client(id);
	}
	return;
}
**/



static struct ibv_pd * rc_get_pd(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	return ctx->pd;
}

static void _build_params(struct rdma_conn_param *params)
{
	memset(params, 0, sizeof(*params));

	params->initiator_depth = params->responder_resources = 1;
	params->rnr_retry_count = 7; /* infinite retry */
	params->retry_count = 7;
}

static void _build_context(struct rdma_cm_id *id)
{
	struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
	s_ctx->ibv_ctx = id->verbs;
	TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ibv_ctx));
	TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ibv_ctx));
	TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ibv_ctx, MAX_CONCURRENCY * 2 + 10, NULL, s_ctx->comp_channel, 0));
	TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));
	id->context = (void*)s_ctx;
	/*
	if (is_server)
	{
		_rdma_thread_pack_* rtp = get_new_thread_pack(id, nit);
		TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, recv_poll_cq, (void*)rtp));
		id->context = (void*)s_ctx;
	}
	**/
}

static void _build_qp_attr(struct ibv_qp_init_attr *qp_attr, struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	memset(qp_attr, 0, sizeof(*qp_attr));
	qp_attr->send_cq = ctx->cq;
	qp_attr->recv_cq = ctx->cq;
	qp_attr->qp_type = IBV_QPT_RC;

	qp_attr->cap.max_send_wr = MAX_CONCURRENCY + 2;
	qp_attr->cap.max_recv_wr = MAX_CONCURRENCY + 2;
	qp_attr->cap.max_send_sge = 1;
	qp_attr->cap.max_recv_sge = 1;
}

static void _build_connection(struct rdma_cm_id *id)
{
	struct ibv_qp_init_attr qp_attr;
	_build_context(id);
	_build_qp_attr(&qp_attr, id);

	struct context *ctx = (struct context *)id->context;
	TEST_NZ(rdma_create_qp(id, ctx->pd, &qp_attr));
}




void _write_remote(struct rdma_cm_id *id, uint32_t len, uint32_t index)
{
	struct context *new_ctx = (struct context *)id->context;

	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	memset(&wr, 0, sizeof(wr));

	wr.wr_id = (uintptr_t)id;

	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = index;//htonl(index);
	wr.wr.rdma.remote_addr = new_ctx->peer_addr[index];
	wr.wr.rdma.rkey = new_ctx->peer_rkey[index];

	if (len)
	{
		wr.sg_list = &sge;
		wr.num_sge = 1;

		sge.addr = (uintptr_t)new_ctx->buffer[index];
		sge.length = len;
		sge.lkey = new_ctx->buffer_mr[index]->lkey;
	}

	TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
}

void _post_receive(struct rdma_cm_id *id, uint32_t index)
{
	struct ibv_recv_wr wr, *bad_wr = NULL;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uint64_t)id;
	wr.sg_list = NULL;
	wr.num_sge = 0;

	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));
}
void _ack_remote(struct rdma_cm_id *id, uint32_t index)
{
	struct context *new_ctx = (struct context *)id->context;

	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	memset(&wr, 0, sizeof(wr));

	wr.wr_id = (uintptr_t)id;

	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = index;//htonl(index);
	wr.wr.rdma.remote_addr = new_ctx->peer_addr[index];
	wr.wr.rdma.rkey = new_ctx->peer_rkey[index];

	new_ctx->ack[index]->index = index;

	{
		wr.sg_list = &sge;
		wr.num_sge = 1;

		sge.addr = (uintptr_t)new_ctx->ack[index];
		sge.length = sizeof(_ack_);
		sge.lkey = new_ctx->ack_mr[index]->lkey;
	}

	TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
}



static void _on_pre_conn(struct rdma_cm_id *id)
{
	struct context *new_ctx = (struct context *)id->context;


	for (int index = 0; index < MAX_CONCURRENCY; index++)
	{
		posix_memalign((void **)(&(new_ctx->buffer[index])), sysconf(_SC_PAGESIZE), BUFFER_SIZE);
		TEST_Z(new_ctx->buffer_mr[index] = ibv_reg_mr(rc_get_pd(id), new_ctx->buffer[index], BUFFER_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
		printf("buffer %d :%p\n", index, new_ctx->buffer_mr[index]->addr);

		posix_memalign((void **)(&(new_ctx->ack[index])), sysconf(_SC_PAGESIZE), sizeof(_ack_));
		TEST_Z(new_ctx->ack_mr[index] = ibv_reg_mr(rc_get_pd(id), new_ctx->ack[index],
		                                sizeof(_ack_), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
		printf("ack %d :%p\n", index, new_ctx->ack_mr[index]->addr);
	}
	log_info("register %d tx_buffer and rx_ack\n", MAX_CONCURRENCY);

	{
		posix_memalign((void **)(&(new_ctx->k_exch[0])), sysconf(_SC_PAGESIZE), sizeof(_key_exch));
		TEST_Z(new_ctx->k_exch_mr[0] = ibv_reg_mr(rc_get_pd(id), new_ctx->k_exch[0], sizeof(_key_exch), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

		posix_memalign((void **)(&(new_ctx->k_exch[1])), sysconf(_SC_PAGESIZE), sizeof(_key_exch));
		TEST_Z(new_ctx->k_exch_mr[1] = ibv_reg_mr(rc_get_pd(id), new_ctx->k_exch[1], sizeof(_key_exch), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

	}
	log_info("register rx_k_exch (index:0) and tx_k_exch (index:1)\n");

	struct ibv_recv_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	memset(&wr, 0, sizeof(wr));

	wr.wr_id = (uintptr_t)id;
	wr.sg_list = &sge;
	wr.num_sge = 1;

	sge.addr = (uintptr_t)(new_ctx->k_exch[1]);
	sge.length = sizeof(_key_exch);
	sge.lkey = new_ctx->k_exch_mr[1]->lkey;



	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));


	for (uint32_t index = 0; index < MAX_CONCURRENCY; index++)
	{
		//log_info("post recv index : %u\n", index);
		_post_receive(id, index);
	}
}

static void _on_connection(struct rdma_cm_id *id, bool is_server)
{
	struct context *new_ctx = (struct context *)id->context;

	int index = 0;

	new_ctx->k_exch[0]->id = MSG_MR;
	if (is_server)
		new_ctx->k_exch[0]->md5 = 6666;
	else
		new_ctx->k_exch[0]->md5 = 5555;

	log_info("k_exch md5 is %llu\n", new_ctx->k_exch[0]->md5);
	if (is_server)
	{
		for (index = 0; index < MAX_CONCURRENCY; index++)
		{
			new_ctx->k_exch[0]->key_info[index].addr = (uintptr_t)(new_ctx->buffer_mr[index]->addr);
			new_ctx->k_exch[0]->key_info[index].rkey = (new_ctx->buffer_mr[index]->rkey);
		}

	}
	else
	{
		for (index = 0; index < MAX_CONCURRENCY; index++)
		{
			new_ctx->k_exch[0]->key_info[index].addr = (uintptr_t)(new_ctx->ack_mr[index]->addr);
			new_ctx->k_exch[0]->key_info[index].rkey = (new_ctx->ack_mr[index]->rkey);
		}

	}


	//send to myself info to peer
	{
		struct ibv_send_wr wr, *bad_wr = NULL;
		struct ibv_sge sge;

		memset(&wr, 0, sizeof(wr));

		wr.wr_id = (uintptr_t)id;
		wr.opcode = IBV_WR_SEND;
		wr.sg_list = &sge;
		wr.num_sge = 1;
		wr.send_flags = IBV_SEND_SIGNALED;

		sge.addr = (uintptr_t)(new_ctx->k_exch[0]);
		sge.length = sizeof(_key_exch);
		sge.lkey = new_ctx->k_exch_mr[0]->lkey;

		TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
	}
	log_info("share my registed mem rx_buffer for peer write to\n");
}

static void _on_disconnect(struct rdma_cm_id *id)
{
	struct context *new_ctx = (struct context *)id->context;

	for (int index = 0; index < MAX_CONCURRENCY; index++)
	{
		ibv_dereg_mr(new_ctx->buffer_mr[index]);
		ibv_dereg_mr(new_ctx->ack_mr[index]);

		free(new_ctx->buffer[index]);
		free(new_ctx->ack[index]);
	}

	{
		ibv_dereg_mr(new_ctx->k_exch_mr[0]);
		ibv_dereg_mr(new_ctx->k_exch_mr[1]);

		free(new_ctx->k_exch[0]);
		free(new_ctx->k_exch[1]);
	}

	free(new_ctx);
}


struct rdma_cm_id* server_wait4conn(struct rdma_event_channel *event_channel)
{
	//bcube_struct& bs = bgs.bcube_s;
	struct rdma_cm_event *event = NULL;
	struct rdma_conn_param cm_params;
	int connecting_client_cnt = 0;
	int client_counts = 1;
	printf("server is inited done (RDMA), waiting for %d client connecting....:)\n", client_counts);
	_build_params(&cm_params);
	std::vector<node_item*> recv_chain;
	struct rdma_cm_id*recv_rdma_cm_id = NULL;
	struct rdma_cm_event event_copy;

	while (rdma_get_cm_event(event_channel, &event) == 0)
	{

		memcpy(&event_copy, event, sizeof(*event));
		rdma_ack_cm_event(event);

		if (event_copy.event == RDMA_CM_EVENT_CONNECT_REQUEST)
		{
			//node_item* nit = get_new_node();
			//recv_chain.push_back(nitn			printf("recv connection Comes\n");
			//build_connection(event_copy.id, IS_SERVER, nit);
			_build_connection(event_copy.id);
			//on_pre_conn(event_copy.id, IS_SERVER);

			_on_pre_conn(event_copy.id);
			TEST_NZ(rdma_accept(event_copy.id, &cm_params));
		}
		else if (event_copy.event == RDMA_CM_EVENT_ESTABLISHED)
		{
			//on_connection(event_copy.id);
			_on_connection(event_copy.id, true);
			//bs.recv_rdma_cm_id.push_back(event_copy.id);
			recv_rdma_cm_id = event_copy.id;
			struct sockaddr_in* client_addr = (struct sockaddr_in *)rdma_get_peer_addr(event_copy.id);
			printf("client[%s,%d] is connecting (RDMA) now... \n", inet_ntoa(client_addr->sin_addr), client_addr->sin_port);
			connecting_client_cnt++;
			if (connecting_client_cnt == client_counts)
				break;
		}
		else if (event_copy.event == RDMA_CM_EVENT_DISCONNECTED)
		{
			rdma_destroy_qp(event_copy.id);
			//on_disconnect(event_copy.id);
			_on_disconnect(event_copy.id);
			rdma_destroy_id(event_copy.id);
			connecting_client_cnt--;
			if (connecting_client_cnt == 0)
				break;
		}
		else
		{
			rc_die("unknown event server\n");
		}
	}
	printf("%d clients have connected to my node (RDMA), ready to receiving loops\n", client_counts);
	return event_copy.id;
}

struct rdma_event_channel* rdma_server_init(int local_port)
{
	int init_loops = 0;
	struct sockaddr_in sin;
	printf("init a server (RDMA)....\n");
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;/*ipv4*/
	sin.sin_port = htons(local_port);/*server listen public ports*/
	sin.sin_addr.s_addr = INADDR_ANY;/*listen any connects*/

	struct rdma_event_channel *event_channel = NULL;
	struct rdma_cm_id *listener = NULL;
	TEST_Z(event_channel = rdma_create_event_channel());
	TEST_NZ(rdma_create_id(event_channel, &listener, NULL, RDMA_PS_TCP));

	while (rdma_bind_addr(listener, (struct sockaddr *)&sin))
	{
		std::cerr << "server init failed (RDMA): error in bind socket, will try it again in 2 seconds..." << std::endl;
		if (init_loops > 10)
		{
			rdma_destroy_id(listener);
			rdma_destroy_event_channel(event_channel);
			exit(-1);
		}
		std::this_thread::sleep_for(std::chrono::seconds(2));
		init_loops++;
	}

	int client_counts = 1;
	if (rdma_listen(listener, client_counts))
	{
		std::cerr << "server init failed (RDMA): error in server listening" << std::endl;
		rdma_destroy_id(listener);
		rdma_destroy_event_channel(event_channel);
		exit(-1);
	}
	printf("Listening port =%d\n", local_port );
	//bcube_gs.bg_thread.push_back(std::thread(recv_RDMA, std::ref(bcube_gs)));

	std::this_thread::sleep_for(std::chrono::seconds(1));
	return  event_channel;
}



struct rdma_cm_id* rdma_client_init_connection(char* local_ip, char* remote_ip, int remote_port)
{
	//std::cout << "client inited (RDMA) start" << std::endl;
	printf("client inited (RDMA) start\n");
	printf("local_ip=%s remote_ip=%s remote_port=%d\n", local_ip, remote_ip, remote_port);

	struct rdma_cm_id *conn = NULL;
	struct rdma_event_channel *ec = NULL;
	//std::string local_eth = bs.local_info.myip[lev];/*get each lev ip*/
	struct sockaddr_in ser_in, local_in;/*server ip and local ip*/
	int connect_count = 0;
	memset(&ser_in, 0, sizeof(ser_in));
	memset(&local_in, 0, sizeof(local_in));

	/*bind remote socket*/
	ser_in.sin_family = AF_INET;
	ser_in.sin_port = htons(remote_port);/*connect to public port remote*/
	inet_pton(AF_INET, remote_ip, &ser_in.sin_addr);

	/*bind local part*/
	local_in.sin_family = AF_INET;
	//std::cout << local_eth.c_str() << "----->" << bs.neighbor_info[lev][index].ip.c_str() << std::endl;
	inet_pton(AF_INET, local_ip, &local_in.sin_addr);

	TEST_Z(ec = rdma_create_event_channel());
	TEST_NZ(rdma_create_id(ec, &conn, NULL, RDMA_PS_TCP));
	TEST_NZ(rdma_resolve_addr(conn, (struct sockaddr*)(&local_in), (struct sockaddr*)(&ser_in), TIMEOUT_IN_MS));

	struct rdma_cm_event *event = NULL;
	struct rdma_conn_param cm_params;
	printf("before build_params\n");
	_build_params(&cm_params);




	while (rdma_get_cm_event(ec, &event) == 0)
	{
		struct rdma_cm_event event_copy;
		memcpy(&event_copy, event, sizeof(*event));
		rdma_ack_cm_event(event);
		if (event_copy.event == RDMA_CM_EVENT_ADDR_RESOLVED)
		{
			printf("RDMA_CM_EVENT_ADDR_RESOLVED\n");
			_build_connection(event_copy.id);
			//on_pre_conn(event_copy.id, IS_CLIENT);
			_on_pre_conn(event_copy.id);

			TEST_NZ(rdma_resolve_route(event_copy.id, TIMEOUT_IN_MS));
		}
		else if (event_copy.event == RDMA_CM_EVENT_ROUTE_RESOLVED)
		{
			printf("RDMA_CM_EVENT_ROUTE_RESOLVED\n");
			TEST_NZ(rdma_connect(event_copy.id, &cm_params));
		}
		else if (event_copy.event == RDMA_CM_EVENT_ESTABLISHED)
		{
			printf("RDMA_CM_EVENT_ESTABLISHED\n");
			struct context *ctx = (struct context *)event_copy.id->context;
			//TEST_NZ(pthread_create(&ctx->cq_poller_thread, NULL, send_poll_cq, event_copy.id));
			std::cout << local_ip << " has connected to server[ " << remote_ip << " , " << remote_port << " ]" << std::endl;
			return event_copy.id;
			//break;
		}
		else if (event_copy.event == RDMA_CM_EVENT_REJECTED)
		{
			printf("RDMA_CM_EVENT_REJECTED\n");
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			connect_count++;
			_on_disconnect(event_copy.id);
			rdma_destroy_qp(event_copy.id);
			rdma_destroy_id(event_copy.id);
			rdma_destroy_event_channel(ec);
			if (connect_count > 10 * 600)/*after 600 seconds, it will exit.*/
			{
				std::cerr << 600 << "seconds is passed, error in connect to server"  << ", check your network condition" << std::endl;
				exit(-1);
			}
			else
			{
				TEST_Z(ec = rdma_create_event_channel());
				TEST_NZ(rdma_create_id(ec, &conn, NULL, RDMA_PS_TCP));
				TEST_NZ(rdma_resolve_addr(conn, (struct sockaddr*)(&local_in), (struct sockaddr*)(&ser_in), TIMEOUT_IN_MS));
			}
		}
		else
		{
			printf("event = %d\n", event_copy.event);
			rc_die("unknown event client\n");
		}
	}
	//bs.topo[lev][bs.neighbor_info[lev][index].node_index].send_rdma_event_channel = ec;
	//bs.topo[lev][bs.neighbor_info[lev][index].node_index].send_rdma_cm_id = conn;
	//bs.neighbor_info[lev][index].send_rdma_event_channel = ec;
	//bs.neighbor_info[lev][index].send_rdma_cm_id = conn;

	rdma_client_establisted = true;
	std::cout << "client inited done" << std::endl;
	return NULL;
}

void test_rdma_header()
{
	printf("Hello\n");
}






#endif