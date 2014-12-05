#include "omp_transfer.h"

hmpi_omp_mode __hmpi_omp_policy = NONE ;
uint64_t* omp_mpi_time_list ;
int core_allocation_rank;

void _hmpi_omp_init(HMPI_Comm comm){
    int local_ranks = comm->node_size ;

#ifdef LB_POLICY_CONSTANT
    __hmpi_omp_policy = CONSTANT;
    //use 3/4rd of cores all the time
    core_allocation_rank = OMP_TOTAL_CORES;
#endif

#ifdef LB_POLICY_DYNAMIC
    __hmpi_omp_policy = STATIC;
    core_allocation_rank = OMP_TOTAL_CORES/local_ranks;
#endif

#ifdef LB_POLICY_SPECULATIVE
    __hmpi_omp_policy = SPECULATIVE;
    core_allocation_rank = DEFAULT_OMP_ALLOCATION;
    if(comm->node_rank == 0) {
        int i = 0 ;
        //One rank per node allocates shared send request lists.
//        omp_mpi_time_list = (uint64_t*) malloc(sizeof(uint64_t) * local_ranks);
        omp_mpi_time_list = (uint64_t*) MALLOC(uint64_t, local_ranks);
        for(i = 0 ; i < local_ranks ; i++){
            omp_mpi_time_list[i] = 0 ;
        }
    }
    MPI_Bcast(&omp_mpi_time_list, 1, MPI_LONG, 0, HMPI_COMM_WORLD->node_comm);
#endif


}
