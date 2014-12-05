#include "omp_transfer.h"
#ifdef __MIC__
#include <immintrin.h>
#endif

inline void transfer_omp_loop_reg(uintptr_t rbuf, uintptr_t sbuf, size_t size, HMPI_Request recv_req, HMPI_Request send_req){
    int i = 0;
#pragma omp parallel
    {
#pragma omp for
        for (i = 0; i < size; i++) {
            ((char *) rbuf)[i] = ((char *) sbuf)[i];
        }
    }

}

inline void transfer_omp_loop_nontemp(uintptr_t rbuf, uintptr_t sbuf, size_t size, HMPI_Request recv_req, HMPI_Request send_req){
    int N_DOUBLES_PER_BLOCK = (64/sizeof(char)) ;
    size_t total = size / 64 ;
    int i = 0;
//#pragma vector nontemporal
#pragma omp parallel for
    for (i = 0; i < total; i++) {
        __m512d v_b = _mm512_load_pd(sbuf+ N_DOUBLES_PER_BLOCK*i);
        _mm512_storenrngo_pd(rbuf+ N_DOUBLES_PER_BLOCK*i, v_b);
    }

}

void transfer_omp_loop(uintptr_t rbuf, uintptr_t sbuf, size_t size, HMPI_Request recv_req, HMPI_Request send_req){

//set number of threads dynamically
//load balance ploicy - depends on number of ranks
// OR speculative execution (running time/MPI time

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(core_allocation_rank); // Use 4 threads for all consecutive parallel regions

#ifdef LB_POLICY_SPECULATIVE
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

        int i = 0;
    #pragma omp for private(i)
        for (i = 0; i < size; i++) {
            ((char *) rbuf)[i] = ((char *) sbuf)[i];
        }
    }
#else
    #ifndef MIC_HMPI_OMP_NON_TEMPORAL
        transfer_omp_loop_reg(rbuf, sbuf, size, recv_req, send_req);
    #else
        transfer_omp_loop_nontemp(rbuf, sbuf, size, recv_req, send_req);
    #endif
#endif

}

/*
void transfer_omp_loop(uintptr_t rbuf, uintptr_t sbuf, size_t size, HMPI_Request recv_req, HMPI_Request send_req){
	int N_DOUBLES_PER_BLOCK = (64/sizeof(char)) ;
        size_t total = size / 64 ; 
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
    	omp_set_num_threads(120); 
	int i = 0;
#pragma omp parallel
    {
        // Work with teh set structure
        cpu_set_t set;
        CPU_ZERO(&set);
        //bind allocated cores to threads
        //int k = 0 ;
        // for(k = core_allocation_start ; k < core_allocation_end ; k++)
          //  CPU_SET(k, &set);
	//
	CPU_SET(HMPI_COMM_WORLD->node_rank+1, &set);
	//printf("comm rank : %d \n", HMPI_COMM_WORLD->node_rank);
        //actual binding/affinity setting take place here
        pid_t tid = (pid_t) syscall(SYS_gettid);
        //sched_setaffinity(tid, sizeof(set), &set);
        //sched_getaffinity(tid, sizeof(set), &set);
	//int rank = 0;
	//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//printf("Thread %d, tid %d, affinity %lu  node_rank : %d \n", omp_get_thread_num(), tid, set, rank);
//#pragma vector nontemporal
#pragma omp for
	for (i = 0; i < total; i++) {
		    __m512d v_b = _mm512_load_pd(sbuf+ N_DOUBLES_PER_BLOCK*i);
     		    _mm512_storenrngo_pd(rbuf+ N_DOUBLES_PER_BLOCK*i, v_b);
	}
}

}*/
/*
void transfer_omp_loop(uintptr_t rbuf, uintptr_t sbuf, size_t size, HMPI_Request recv_req, HMPI_Request send_req){
	int N_DOUBLES_PER_BLOCK = (64/sizeof(char)) ;
        size_t total = size / 64 ; 
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
    	omp_set_num_threads(120); 
	int i = 0;
//#pragma vector nontemporal
#pragma omp parallel for
	for (i = 0; i < total; i++) {
		    __m512d v_b = _mm512_load_pd(sbuf+ N_DOUBLES_PER_BLOCK*i);
     		    _mm512_storenrngo_pd(rbuf+ N_DOUBLES_PER_BLOCK*i, v_b);
	}

}*/

/*
void transfer_omp_loop(uintptr_t rbuf, uintptr_t sbuf, size_t size, HMPI_Request recv_req, HMPI_Request send_req){

//set number of threads dynamically
//load balance ploicy - depends on number of ranks
// OR speculative execution (running time/MPI time

    //omp_set_dynamic(0);     // Explicitly disable dynamic teams
    //omp_set_num_threads(core_allocation_rank); // Use 4 threads for all consecutive parallel regions
    omp_set_num_threads(10); // Use 4 threads for all consecutive parallel regions

int i = 0 ;
#pragma vector nontemporal
#pragma simd
#pragma omp parallel for
        for (i = 0; i < size; i++) {
            ((char *) rbuf)[i] = ((char *) sbuf)[i];
        }

//transfer by characters

#pragma omp parallel
    {
#if HMPI_OMP_AFFINITY_AWARE == 1
        // Work with teh set structure
        /cpu_set_t set;
        CPU_ZERO(&set);
        //bind allocated cores to threads
        int k = 0 ;
        for(k = core_allocation_start ; k < core_allocation_end ; k++)
            CPU_SET(k, &set);

        //actual binding/affinity setting take place here
        pid_t tid = (pid_t) syscall(SYS_gettid);
        sched_setaffinity(tid, sizeof(set), &set);

#endif

        int i = 0;
#pragma omp for private(i)
        for (i = 0; i < size; i++) {
            ((char *) rbuf)[i] = ((char *) sbuf)[i];
        }
    }


}        */



