#ifndef RDMA_MESSAGES_H
#define RDMA_MESSAGES_H

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <rdma/rdma_cma.h>

const char *DEFAULT_PORT = "12345";
const size_t BUFFER_SIZE = 10 * 1024 * 1024;

#define TIMEOUT_IN_MS 500


#define TEST_NZ(x) do { if ( (x)) rc_die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) rc_die("error: " #x " failed (returned zero/null)."); } while (0)

typedef void (*pre_conn_cb_fn)(struct rdma_cm_id *id);
typedef void (*connect_cb_fn)(struct rdma_cm_id *id);
typedef void (*completion_cb_fn)(struct ibv_wc *wc);
typedef void (*disconnect_cb_fn)(struct rdma_cm_id *id);


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

#endif
