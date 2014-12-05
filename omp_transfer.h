/* Copyright (c) 2010-2013 The Trustees of Indiana University.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * - Neither the Indiana University nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _OMP_TMODE_HEADER
#define _OMP_TMODE_HEADER

#ifndef HMPI_INTERNAL
#define HMPI_INTERNAL 1
#endif

#include <stdio.h>
#include <stdlib.h>
#include "hmpi.h"
#include <omp.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <sched.h>

#define OMP_TOTAL_CORES  120
#define DEFAULT_OMP_ALLOCATION  5

#define HMPI_ATOMIC_ADD(p, v) (uint64_t)__sync_add_and_fetch(p, v)

#define HMPI_ATOMIC_GET(p) (uint64_t)__sync_add_and_fetch(p, 0.0)
/*
* Three different load balancing policies
* CONSTANT - use constant number of cores for all transfers
*
* DYNAMIC - static load balancing based on number of ranks. Each rank recieves predefined number of threads attached
*          to cores. number of allocated cores per rank is caluclated on OMP_TOTAL_CORES/R
*
* SPECULATIVE - speculate available number of cores per transfer based on function time spent and size. f(t, s)
* */

//#define HMPI_OMP_MODE 0xFF

#ifdef HMPI_OMP_MODE
#define LB_POLICY_CONSTANT 1000
#endif

//#define LB_POLICY_DYNAMIC 1001
//#define LB_POLICY_SPECULATIVE 1002

#define HMPI_OMP_AFFINITY_AWARE 1

typedef enum hmpi_omp_mode {
    NONE,
    CONSTANT,
    STATIC,
    SPECULATIVE
}hmpi_omp_mode ;

/*
* shared data structure/storage for
* speculative measuremnts of mpi_time
* */
extern uint64_t* omp_mpi_time_list ;

extern hmpi_omp_mode __hmpi_omp_policy;

extern int core_allocation_rank ;

#if HMPI_OMP_AFFINITY_AWARE == 1
extern int core_allocation_start ;
extern int core_allocation_end ;
#endif


void _hmpi_omp_init(HMPI_Comm comm);

void transfer_omp_loop(uintptr_t rbuf, uintptr_t sbuf, size_t size, HMPI_Request recv_req, HMPI_Request send_req);

void profile_omp_loop(uint64_t dt, int rank);

uint64_t t_useconds();

#endif

