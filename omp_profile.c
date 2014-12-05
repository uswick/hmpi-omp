#include "omp_transfer.h"

#if HMPI_OMP_AFFINITY_AWARE == 1
int  core_allocation_start = 0;
int core_allocation_end = 0 ;
#endif

void profile_omp_loop(uint64_t dt_ns, int rank){
    if(__hmpi_omp_policy == SPECULATIVE) {
        uint64_t curr_rank_mpi_time = HMPI_ATOMIC_ADD(&omp_mpi_time_list[rank], dt_ns);
        int total = HMPI_COMM_WORLD->node_size;

        //accumalate total mpi time at this point in time
        int r = 0;
        uint64_t t_mpi = 0;
        uint64_t t_mpi_before_rank = 0;
        uint64_t t_mpi_after_rank = 0;
        for (r = 0; r < total; r++) {
            t_mpi += HMPI_ATOMIC_GET(&omp_mpi_time_list[r]);
#if HMPI_OMP_AFFINITY_AWARE == 1
            if (r < rank) {
                t_mpi_before_rank += HMPI_ATOMIC_GET(&omp_mpi_time_list[r]);
            } else if (r > rank) {
                t_mpi_after_rank += HMPI_ATOMIC_GET(&omp_mpi_time_list[r]);
            }
#endif
        }

        //calculate allocation based speculating time
        //we assume time allocated propotianal to allocated cores
        // larger the time spent more we need to allocate
        core_allocation_rank = OMP_TOTAL_CORES * curr_rank_mpi_time / t_mpi;



#if HMPI_OMP_AFFINITY_AWARE == 1
        //starting core index prior to this acllocation
        core_allocation_start = OMP_TOTAL_CORES * t_mpi_before_rank / t_mpi;

        //starting core index allocated after this set
//        core_allocation_end = OMP_TOTAL_CORES - (OMP_TOTAL_CORES * t_mpi_after_rank / t_mpi) - 1;
        core_allocation_end = core_allocation_start + core_allocation_rank;
#endif

        if (core_allocation_rank <= 0) {
            core_allocation_rank = DEFAULT_OMP_ALLOCATION;
            core_allocation_end = core_allocation_start + DEFAULT_OMP_ALLOCATION;
        }

#if HMPI_OMP_AFFINITY_AWARE == 1
        //sanity check
        if (core_allocation_end <= 0) {
            core_allocation_end = core_allocation_start + core_allocation_rank;
        }else if(core_allocation_end > OMP_TOTAL_CORES){
            core_allocation_end = OMP_TOTAL_CORES;
        }
#endif
    }
#if HMPI_OMP_AFFINITY_AWARE == 1
    else if (__hmpi_omp_policy == STATIC){
        //linear static allocation if CPU_affinity is used
        core_allocation_start = core_allocation_rank * rank;
        core_allocation_end = core_allocation_start + core_allocation_rank;
    }
#endif
#if HMPI_OMP_AFFINITY_AWARE == 1
    else if (__hmpi_omp_policy == CONSTANT){
        //linear static allocation if CPU_affinity is used
        core_allocation_start = 0;
        core_allocation_end = core_allocation_start + core_allocation_rank;
    }
#endif
}

#include <sys/time.h>

uint64_t t_useconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( tp.tv_sec * 1e6 + tp.tv_usec );
}
