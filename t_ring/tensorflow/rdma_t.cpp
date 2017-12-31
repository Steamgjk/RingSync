#include "rdma_t.h"
#include <atomic>
#include <cstring>

#if HAVE_RDMA

#include <errno.h>
#include <iostream>
#include <arpa/inet.h>

#define IS_SERVER true
#define IS_CLIENT false

extern std::atomic_bool server_establisted;
extern std::atomic_bool client_establisted;
extern void show_msg(void*);

void rc_die(const char* reason)
{
	fprintf(stderr, "%s\n", reason);
	exit(-1);
}

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

static void send_tensor(struct rdma_cm_id *id, char* buff, uint32_t len)
{
	struct context *ctx = (struct context *)id->context;
	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;
	if (!buff)throw std::runtime_error("send buf can not be empty!");


	std::memcpy(ctx->buffer, buff, len);
	delete[] buff;

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

static void post_receive_client(struct rdma_cm_id *id)
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


static void* recv_data(struct ibv_wc* wc)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;
	void* _data = NULL;
	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)//as server only
	{
		uint32_t size = ntohl(wc->imm_data);
		//struct sockaddr_in* client_addr = (struct sockaddr_in *)rdma_get_peer_addr(id);
		_data = new char[size];
		std::memcpy(_data, ctx->buffer, size);
		post_receive_server(id);
		ctx->msg->id = MSG_READY;
		send_message(id);
	}
	else
	{
		std::cout << "recv op code is " << wc->opcode << std::endl;
	}
	return _data;
}

static void* send_data(struct ibv_wc* wc, void* data, int len)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;

	if (wc->opcode & IBV_WC_RECV)
	{
		if (ctx->msg->id == MSG_MR)
		{
			ctx->peer_addr = ctx->msg->data.mr.addr;
			ctx->peer_rkey = ctx->msg->data.mr.rkey;
			printf("received remote memory address and key\n");
			ctx->remote_idle = true;
			send_tensor(id, (char*)data, len);
		}
		else if (ctx->msg->id == MSG_DONE)
		{
			printf("received DONE, disconnecting\n");
			rdma_disconnect(id);
			return NULL;
		}
		else if (ctx->msg->id == MSG_READY)
		{
			ctx->remote_idle = true;
			send_tensor(id, (char*)data, len);
		}
		post_receive_client(id);
	}
	else
	{
		std::cout << "send op code is " << wc->opcode << std::endl;
	}
	return NULL;
}

void rcv_poll_cq(void *tmp_id, _recv_chain* chain_header)
{
	struct ibv_cq *cq = NULL;
	struct ibv_wc wc;
	struct rdma_cm_id *id = (struct rdma_cm_id *)tmp_id;
	struct context *ctx = (struct context *)id->context;
	void *ev_ctx = NULL;

	_recv_chain* rcv_tail = chain_header;

	while (true)
	{
		TEST_NZ(ibv_get_cq_event(ctx->comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));

		while (ibv_poll_cq(cq, 1, &wc))
		{
			if (wc.status == IBV_WC_SUCCESS)
			{
				//if (wc.opcode == IBV_WC_RECV)
				{
					auto recved_ptr = recv_data(&wc);
					if (!recved_ptr)continue;

					auto tp_node = new _recv_chain;
					tp_node->data_ptr = recved_ptr;
					tp_node->next = NULL;
					rcv_tail->next = tp_node;
					rcv_tail = tp_node;
				}
			}

			else
			{
				printf("\nwc = %s\n", ibv_wc_status_str(wc.status));
				rc_die("poll_cq: status is not IBV_WC_SUCCESS");
			}
		}
	}
	return ;
}

void send_poll_cq(void * tmp_id, _recv_chain* chain_header)
{
	struct ibv_cq *cq = NULL;
	struct ibv_wc wc;
	struct rdma_cm_id *id = (struct rdma_cm_id *)tmp_id;
	struct context *ctx = (struct context *)id->context;
	void *ev_ctx = NULL;

	_recv_chain* rcv_header = chain_header;
	//std::this_thread::sleep_for(std::chrono::seconds(5));

	while (true)
	{
		TEST_NZ(ibv_get_cq_event(ctx->comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));
		while (!(rcv_header->next))//no data is ready... will sleep
			std::this_thread::sleep_for(std::chrono::nanoseconds(10));

		while (ibv_poll_cq(cq, 1, &wc))
		{
			if (wc.status == IBV_WC_SUCCESS)
			{
				//if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM)
				{
					auto tp_node = rcv_header->next;
					send_data(&wc, tp_node->data_ptr, tp_node->data_len);
					delete rcv_header;
					rcv_header = tp_node;
				}
			}
			else
			{
				printf("\nwc = %s\n", ibv_wc_status_str(wc.status));
				rc_die("poll_cq: status is not IBV_WC_SUCCESS");
			}
		}
	}
	return ;
}

struct ibv_pd * rc_get_pd(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	return ctx->pd;
}

void build_params(struct rdma_conn_param *params)
{
	memset(params, 0, sizeof(*params));
	params->initiator_depth = params->responder_resources = 1;
	params->rnr_retry_count = params->retry_count = 7;/* infinite retry */
}

void build_context(struct rdma_cm_id *id, bool is_server, _recv_chain* chain_header)
{
	struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
	s_ctx->ibv_ctx = id->verbs;
	TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ibv_ctx));
	TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ibv_ctx));
	TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ibv_ctx, MIN_CQE, NULL, s_ctx->comp_channel, 0));
	TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));
	id->context = (void*)s_ctx;
	if (is_server)
	{
		s_ctx->cq_poller_thread = std::thread(rcv_poll_cq, id, chain_header);/*create recv threads*/
		id->context = (void*)s_ctx;
	}
}

void build_qp_attr(struct ibv_qp_init_attr *qp_attr, struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	memset(qp_attr, 0, sizeof(*qp_attr));
	qp_attr->send_cq = ctx->cq;
	qp_attr->recv_cq = ctx->cq;
	qp_attr->qp_type = IBV_QPT_RC;

	qp_attr->cap.max_send_wr = 10;
	qp_attr->cap.max_recv_wr = 10;
	qp_attr->cap.max_send_sge = 1;
	qp_attr->cap.max_recv_sge = 1;
}

void build_connection(struct rdma_cm_id *id, bool is_server, _recv_chain* chain_header)
{
	struct ibv_qp_init_attr qp_attr;
	build_context(id, is_server, chain_header);
	build_qp_attr(&qp_attr, id);

	struct context *ctx = (struct context *)id->context;
	TEST_NZ(rdma_create_qp(id, ctx->pd, &qp_attr));
}

static void on_pre_conn(struct rdma_cm_id *id, bool is_server)
{
	struct context *ctx = (struct context *)id->context;
	posix_memalign((void **)&ctx->buffer, sysconf(_SC_PAGESIZE), BUFFER_SIZE);
	TEST_Z(ctx->buffer_mr = ibv_reg_mr(rc_get_pd(id), ctx->buffer, BUFFER_SIZE,
	                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

	posix_memalign((void **)&ctx->msg, sysconf(_SC_PAGESIZE), sizeof(*ctx->msg));
	TEST_Z(ctx->msg_mr = ibv_reg_mr(rc_get_pd(id), ctx->msg, sizeof(*ctx->msg),
	                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

	if (is_server)
		post_receive_server(id);
	else
		post_receive_client(id);
}

static void on_connection(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;

	ctx->msg->id = MSG_MR;
	ctx->msg->data.mr.addr = (uintptr_t)ctx->buffer_mr->addr;
	ctx->msg->data.mr.rkey = ctx->buffer_mr->rkey;

	send_message(id);
}

static void on_disconnect(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;

	ibv_dereg_mr(ctx->buffer_mr);
	ibv_dereg_mr(ctx->msg_mr);

	free(ctx->buffer);
	free(ctx->msg);
	free(ctx);
}


//#endif //RDMA
#endif
