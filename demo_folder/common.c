#include "common.h"

//const int TIMEOUT_IN_MS = 500;

struct context
{
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  struct ibv_comp_channel *comp_channel;

  pthread_t cq_poller_thread;
};

static struct context *s_ctx = NULL;


//here



void rc_disconnect(struct rdma_cm_id *id)
{
  rdma_disconnect(id);
}

void rc_die(const char *reason)
{
  fprintf(stderr, "%s\n", reason);
  exit(EXIT_FAILURE);
}



struct ibv_pd * rc_get_pd()
{
  return s_ctx->pd;
}
