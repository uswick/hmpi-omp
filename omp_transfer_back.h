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

#include <stdio.h>
#include <stdlib.h>
#include "hmpi.h"
#include <omp.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <sched.h>

#define OMP_TOTAL_CORES  60
#define DEFAULT_OMP_CORES  5

#define HMPI_ATOMIC_ADD(p, v) (double)__sync_add_and_fetch(p, v)

#define HMPI_ATOMIC_GET(p) (double)__sync_add_and_fetch(p, 0.0)
/*
* Three different load balancing policies
* CONSTANT - use constant number of cores for all transfers
*
* DYNAMIC - static load balancing based on number of ranks. Each rank recieves predefined number of threads attached
*          to cores. number of allocated cores per rank is caluclated on OMP_TOTAL_CORES/R
*
* SPECULATIVE - speculate available number of cores per transfer based on function time spent and size. f(t, s)
* */

#define LB_POLICY_CONSTANT 1000
#define LB_POLICY_STATIC 1001
#define LB_POLICY_SPECULATIVE 1002

#define HMPI_INTERNAL 1
#define HMPI_OMP_AFFINITY_AWARE 1

enum hmpi_omp_mode {
    CONSTANT,
    STATIC,
    SPECULATIVE
} ;

double* omp_mpi_time_list ;

hmpi_omp_mode __hmpi_omp_policy = CONSTANT;

void _hmpi_omp_init(HMPI_Comm comm);

extern int core_allocation_rank ;


#if HMPI_OMP_AFFINITY_AWARE == 1
extern int core_allocation_start ;
extern int core_allocation_end ;

int  core_allocation_rank = 1;
int core_allocation_start = 0 ;
#endif

void _hmpi_omp_init(HMPI_Comm comm){
    int local_ranks = comm->node_size ;

#ifdef LB_POLICY_CONSTANT
    __hmpi_omp_policy = CONSTANT;
    //use 3/4rd of cores all the time
    core_allocation_rank = OMP_TOTAL_CORES*(3/4);
#endif

#ifdef LB_POLICY_STATIC
    __hmpi_omp_policy = STATIC;
    core_allocation_rank = OMP_TOTAL_CORES/local_ranks;
#endif

#ifdef LB_POLICY_SPECULATIVE
    __hmpi_omp_policy = SPECULATIVE;
    core_allocation_rank = DEFAULT_OMP_CORES;
    if(HMPI_COMM_WORLD->node_rank == 0) {
        int i = 0 ;
        //One rank per node allocates shared send request lists.
        omp_mpi_time_list = (double*) MALLOC(double, local_ranks);
        for(i = 0 ; i < local_ranks ; i++){
            omp_mpi_time_list[i] = 0.0 ;
        }
    }
    MPI_Bcast(&omp_mpi_time_list, 1, MPI_LONG, 0, HMPI_COMM_WORLD->node_comm);
#endif

}


void transfer_omp_loop(uintptr_t rbuf, uintptr_t sbuf, size_t size, HMPI_Request recv_req, HMPI_Request send_req);


void transfer_omp_loop(uintptr_t rbuf, uintptr_t sbuf, size_t size, HMPI_Request recv_req, HMPI_Request send_req){

//set number of threads dynamically
//load balance ploicy - depends on number of ranks
// OR speculative execution (running time/MPI time

omp_set_dynamic(0);     // Explicitly disable dynamic teams
omp_set_num_threads(core_allocation_rank); // Use 4 threads for all consecutive parallel regions

//transfer by characters
#pragma omp parallel
{
    #if HMPI_OMP_AFFINITY_AWARE == 1
    // Work with teh set structure
    cpu_set_t set;
    CPU_ZERO(&set);
    //bind allocated cores to threads
    int k = 0 ;
    for(k = core_allocation_start ; k < core_allocation_end ; k++)
        CPU_SET(k, &set);

    //actual binding/affinity setting take place here
    pid_t tid = (pid_t) syscall(SYS_gettid);
    sched_setaffinity(tid, sizeof(set), &set);

    #endif

#pragma omp for
        for (int i = 0; i < size; i++) {
            ((char *) rbuf)[i] = ((char *) sbuf)[i];
        }
}

}

void profile_omp_loop(double dt, int rank){
    double curr_rank_mpi_time = HMPI_ATOMIC_ADD(&omp_mpi_time_list[rank], dt);
    int total = HMPI_COMM_WORLD->node_size;

    //accumalate total mpi time at this point in time
    int r = 0 ;
    double t_mpi = curr_rank_mpi_time ;
    double t_mpi_before_rank = 0.0 ;
    double t_mpi_after_rank = 0.0 ;
    for(r = 0 ; r < total ; r++){
        t_mpi += HMPI_ATOMIC_GET(omp_mpi_time_list[r]) ;
        #if HMPI_OMP_AFFINITY_AWARE == 1
        if(r < rank){
            t_mpi_before_rank += HMPI_ATOMIC_GET(omp_mpi_time_list[r]);
        } else if(r > rank){
            t_mpi_after_rank += HMPI_ATOMIC_GET(omp_mpi_time_list[r]);
        }
        #endif
    }

    //calculate allocation based speculating time
    //we assume time allocated propotianal to allocated cores
    // larger the time spent more we need to allocate
    core_allocation_rank = OMP_TOTAL_CORES * (curr_rank_mpi_time/t_mpi);


#if HMPI_OMP_AFFINITY_AWARE == 1
    //starting core index prior to this acllocation
    core_allocation_start = OMP_TOTAL_CORES * (t_mpi_before_rank/t_mpi);

    //starting core index allocated after this set
    core_allocation_end = OMP_TOTAL_CORES * (t_mpi_after_rank/t_mpi);
#endif

    if(core_allocation_rank <= 0){
        core_allocation_rank = DEFAULT_OMP_CORES;
    }

#if HMPI_OMP_AFFINITY_AWARE == 1
    //sanity check
    if(core_allocation_end == 0){
        core_allocation_end = core_allocation_start + core_allocation_rank;
    }
#endif

}

#endif

