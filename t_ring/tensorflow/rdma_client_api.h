#ifndef __TENSORFLOW_RING_SERVE_RDMA_H__
#define __TENSORFLOW_RING_SERVER_RDMA_H__
#include <vector>
#include <string>
#include <pthread.h>
#include <stdarg.h>
#if HAVE_RDMA
#include <rdma/rdma_cma.h>
#include "rdma_t.h"

void _write_remote(struct rdma_cm_id *id, uint32_t len, uint32_t index);

void _client_post_receive(struct rdma_cm_id *id, uint32_t index);

void _client_on_pre_conn(struct rdma_cm_id *id);

void _client_on_connection(struct rdma_cm_id *id);

void _client_on_completion(struct ibv_wc *wc);

#endif

#endif