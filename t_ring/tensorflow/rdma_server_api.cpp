#ifndef __TENSORFLOW_RING_CLIENT_RDMA_H__
#define __TENSORFLOW_RING_CLIENT_RDMA_H__
#include <vector>
#include <string>
#include <pthread.h>
#include <stdarg.h>
#if HAVE_RDMA
#include <rdma/rdma_cma.h>
#include "rdma_t.h"

void _server_ack_remote(struct rdma_cm_id *id, uint32_t index);

void _server_post_receive(struct rdma_cm_id *id, uint32_t index);

void _server_on_pre_conn(struct rdma_cm_id *id);

void _server_on_completion(struct ibv_wc *wc);

void _server_on_connection(struct rdma_cm_id *id);

void _server_on_disconnect(struct rdma_cm_id *id);


#endif

#endif