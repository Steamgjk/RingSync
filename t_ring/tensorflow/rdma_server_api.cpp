#include "rdma_server_api.h"

extern const size_t BUFFER_SIZE;
void _server_ack_remote(struct rdma_cm_id *id, uint32_t index)
{
	struct conn_context *_ctx = (struct conn_context *)id->context;
	extend_info* new_ctx = &(_ctx->s_new_ctx);

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

void _server_post_receive(struct rdma_cm_id *id, uint32_t index)
{
	struct ibv_recv_wr wr, *bad_wr = NULL;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)id;
	wr.sg_list = NULL;
	wr.num_sge = 0;
	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));
}

void _server_on_pre_conn(struct rdma_cm_id *id)
{
	struct conn_context *_ctx = (struct conn_context *)malloc(sizeof(struct conn_context));

	id->context = _ctx;
	extend_info* new_ctx = &(_ctx->s_new_ctx);
	int index = 0;

	for (index = 0; index < MAX_CONCURRENCY; index++)
	{
		posix_memalign((void **)(&(new_ctx->buffer[index])), sysconf(_SC_PAGESIZE), BUFFER_SIZE);
		TEST_Z(new_ctx->buffer_mr[index] = ibv_reg_mr(rc_get_pd(), new_ctx->buffer[index], BUFFER_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

		posix_memalign((void **)(&(new_ctx->ack[index])), sysconf(_SC_PAGESIZE), sizeof(_ack_));
		TEST_Z(new_ctx->ack_mr[index] = ibv_reg_mr(rc_get_pd(), new_ctx->ack[index],
		                                sizeof(_ack_), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
	}
	log_info("register %d tx_buffer and rx_ack\n", index);

	{
		posix_memalign((void **)(&(new_ctx->k_exch[0])), sysconf(_SC_PAGESIZE), sizeof(_key_exch));
		TEST_Z(new_ctx->k_exch_mr[0] = ibv_reg_mr(rc_get_pd(), new_ctx->k_exch[0], sizeof(_key_exch), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

		posix_memalign((void **)(&(new_ctx->k_exch[1])), sysconf(_SC_PAGESIZE), sizeof(_key_exch));
		TEST_Z(new_ctx->k_exch_mr[1] = ibv_reg_mr(rc_get_pd(), new_ctx->k_exch[1], sizeof(_key_exch), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

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
		log_info("post recv index : %u\n", index);
		_server_post_receive(id, index);
	}
}

void _server_on_completion(struct ibv_wc *wc)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)(wc->wr_id);
	struct conn_context *ctx = (struct conn_context *)id->context;
	extend_info* new_ctx = &(ctx->s_new_ctx);

	//else if (wc->opcode & IBV_WC_RECV)
	if (wc->opcode == IBV_WC_RECV)
	{
		switch (new_ctx->k_exch[1]->id)
		{
		case MSG_MR:
		{
			log_info("recv MD5 is %llu\n", new_ctx->k_exch[1]->md5);
			log_info("received MR, sleep 1000s\n");
			log_info("imm_data is %d\n", wc->imm_data);
			for (int index = 0; index < MAX_CONCURRENCY; index++)
			{
				new_ctx->peer_addr[index] = new_ctx->k_exch[1]->key_info[index].addr;
				new_ctx->peer_rkey[index] = new_ctx->k_exch[1]->key_info[index].rkey;
			}
			//_post_receive(id, 22);
			//sleep(1000);
		} break;
		default:
			break;
		}
		//log_info("nothing to know...");
		//post_receive(id);
	}
	else if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		log_info("recv with IBV_WC_RECV_RDMA_WITH_IMM\n");
		log_info("imm_data is %d\n", wc->imm_data);
		log_info("recv is : %10s\n", new_ctx->buffer[wc->imm_data]);
		_server_ack_remote(id, wc->imm_data);
		_server_post_receive(id, wc->imm_data);

	}
	else
	{
		log_info("wc = %s\n", ibv_wc_status_str(wc->status));
	}
}

void _server_on_connection(struct rdma_cm_id *id)
{
	struct conn_context *_ctx = (struct conn_context *)id->context;
	extend_info* new_ctx = &(_ctx->s_new_ctx);
	int index = 0;

	new_ctx->k_exch[0]->id = MSG_MR;
	new_ctx->k_exch[0]->md5 = 6666;
	log_info("k_exch md5 is %llu\n", new_ctx->k_exch[0]->md5);
	for (index = 0; index < MAX_CONCURRENCY; index++)
	{
		new_ctx->k_exch[0]->key_info[index].addr = (uintptr_t)(new_ctx->buffer_mr[index]->addr);
		new_ctx->k_exch[0]->key_info[index].rkey = (new_ctx->buffer_mr[index]->rkey);
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

void _server_on_disconnect(struct rdma_cm_id *id)
{
	struct conn_context *_ctx = (struct conn_context *)id->context;
	extend_info* new_ctx = &(_ctx->s_new_ctx);
	if (_ctx->fd) fclose(_ctx->fd);

	for (int index = 0; index < MAX_CONCURRENCY; index++)
	{
		ibv_dereg_mr(new_ctx->buffer_mr[index]);
		ibv_dereg_mr(new_ctx->ack_mr[index]);

		free(new_ctx->buffer_mr[index]);
		free(new_ctx->ack_mr[index]);
	}
	//printf("finished transferring %s\n", _ctx->file_name);
	free(_ctx);
}