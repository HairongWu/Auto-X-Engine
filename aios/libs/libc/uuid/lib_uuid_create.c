/****************************************************************************
 * libs/libc/uuid/lib_uuid_create.c
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.  The
 * ASF licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 ****************************************************************************/

/****************************************************************************
 * Included Files
 ****************************************************************************/

#include <sys/random.h>
#include <errno.h>
#include <stdlib.h>
#include <uuid.h>

/****************************************************************************
 * Private Functions
 ****************************************************************************/

static int uuid_getrandom(FAR void *buf, size_t size, int flags)
{
  FAR char *tmp = buf;

  while (size > 0)
    {
      ssize_t ret = getrandom(tmp, size, flags);
      if (ret < 0)
        {
          if (get_errno() == EINTR)
            {
              continue;
            }

          return ret;
        }

      tmp += ret;
      size -= ret;
    }

  return 0;
}

/****************************************************************************
 * Public Functions
 ****************************************************************************/

/****************************************************************************
 * uuid_create_nil
 *
 * Description:
 *   Create an UUID.
 *
 * See also:
 *   http://www.opengroup.org/onlinepubs/009629399/uuid_create.htm
 *
 ****************************************************************************/

void uuid_create(uuid_t *u, uint32_t *status)
{
  int ret;

  ret = uuid_getrandom(u, sizeof(uuid_t), 0);
  if (ret < 0)
    {
      ret = uuid_getrandom(u, sizeof(uuid_t), GRND_RANDOM);
    }

  if (ret < 0)
    {
      unsigned long *beg = (unsigned long *)u;
      unsigned long *end = (unsigned long *)(u + 1);

      while (beg < end)
        {
          *beg++ = rand();
        }
    }

  u->clock_seq_hi_and_reserved &= ~(1 << 6);
  u->clock_seq_hi_and_reserved |= (1 << 7);

  u->time_hi_and_version &= ~(1 << 12);
  u->time_hi_and_version &= ~(1 << 13);
  u->time_hi_and_version |= (1 << 14);
  u->time_hi_and_version &= ~(1 << 15);

  if (status)
    {
      *status = uuid_s_ok;
    }
}
