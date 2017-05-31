#ifdef __cplusplus
extern "C" {
#endif
void MPIWrapper_Initialize();
void MPIWrapper_Finalize();

int MPIWrapper_getRank();
int MPIWrapper_getSize();

int MPIWrapper_MPI_SUM();
int MPIWrapper_MPI_MAX();

void MPIWrapper_Allreduce_double_( double *inarray, int arraylength, int op );
void MPIWrapper_Allreduce_int_( int *inarray, int arraylength, int op );

void MPIWrapper_Reduce_double_( double *inarray, int arraylength, int op );
void MPIWrapper_Reduce_int_( int *inarray, int arraylength, int op );

void MPIWrapper_Bcast_double_( double *inarray, int arraylength );
void MPIWrapper_Bcast_int_( int *inarray, int arraylength );

#ifdef __cplusplus
}
#endif

