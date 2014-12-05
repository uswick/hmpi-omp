#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __MIC__
#include <immintrin.h>
#endif

//int SIZE = 2000000 ;
//int SIZE = 1048576 ;
//int SIZE = 4096 ;
int SIZE = 12288 ;
//int SIZE = 8388608 ;
static int MODE_V_NT = 1001 ;
static int MODE_V_ST = 1002 ;
static int MODE_S_ST = 1003 ;
int mode ;
/*void simple_triad(double*  a,
 double *b,double* c, double* d, int N);

void simple_triad(double* a, 
 double *b, double* c, double* d, int N) 
{ 
int i; 
#pragma omp parallel for
#pragma vector nontemporal 
 for (i=0; i<N; i++) 
 a[i] = b[i] + 0.1 ; 
} */
double  mysecond();

void char_copy(uintptr_t rbuf, uintptr_t sbuf,  size_t size){
	double t = mysecond();
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
    	omp_set_num_threads(120); 
int i = 0;
//#pragma vector nontemporal
#pragma omp parallel for
	for (i = 0; i < size; i++) {
        	    ((char *) rbuf)[i] = ((char *) sbuf)[i];
	}
 	t = mysecond() - t ;
 	printf("AFTER Triad elapsed : %f size bytes : %lu \n", t, size);
}

void char_copy_nontemp(uintptr_t rbuf, uintptr_t sbuf,  size_t size){
	double t = mysecond();
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
 	t = mysecond() - t ;
 	printf("AFTER Triad elapsed : %f size bytes : %lu \n", t, size);
}

void char_copy2(uintptr_t rbuf, uintptr_t sbuf,  size_t size){
	double t = mysecond();
	#pragma vector nontemporal
 	memcpy((void*)rbuf, (void*)sbuf, size);
	t = mysecond() - t ;
 	printf("AFTER Triad elapsed : %f size bytes : %lu \n", t, size);
}

void char_copy3(uintptr_t rbuf, uintptr_t sbuf,  size_t size){
	size_t offset = 0;
        size_t block_size = 12288 ;
	double t = mysecond();
	while(offset < size) {
            size_t left = size - offset;
            memcpy((void*)(rbuf + offset), (void*)(sbuf + offset),
                    (left < block_size ? left : block_size));
	    offset +=block_size;
	}
	t = mysecond() - t ;
 	printf("Blocked copy : AFTER Triad elapsed : %f size bytes : %lu \n", t, size);
}

void simple_triad(double*  a,
 double *b,double* c, double* d, int N);
void simple_triad(double*  a,
 double *b,double* c, double* d, int N);

void simple_triad(double* a, 
 double *b, double* c, double* d, int N) 
{ 
int i; 
int N_DOUBLES_PER_BLOCK = (64/sizeof(double)) ;


__m512d v_b = _mm512_load_pd(b+ N_DOUBLES_PER_BLOCK*0);
double t = mysecond();
#pragma omp parallel for
 for (i=0; i<N; i++) {
     __m512d v_b = _mm512_load_pd(b+ N_DOUBLES_PER_BLOCK*i);
     _mm512_storenrngo_pd(a+ N_DOUBLES_PER_BLOCK*i, v_b);
     //_mm512_storenr_pd(a+ N_DOUBLES_PER_BLOCK*i, v_b);
    //a[i] = b[i]  ;
    /*if (mode == MODE_V_NT){
        __m512 v_b = _mm512_load_ps(b+ N_FLOATS_PER_BLOCK*i);
        _mm512_storenrngo_ps(a+ N_FLOATS_PER_BLOCK*i, v_b);
        //printf("N : %d" , N);
    }
    else if (mode == MODE_V_ST) {
        __m512 v_b = _mm512_load_ps(b+ N_FLOATS_PER_BLOCK*i);
        _mm512_store_ps(a+ N_FLOATS_PER_BLOCK*i, v_b);
        //printf("N : %d" , N);
    } else {
 	a[i] = b[i]  ;
        //printf("N : %d" , N);
    }*/

  } 
 t = mysecond() - t ;

 printf("AFTER Triad elapsed : %f N : %d \n", t, N);
} 

int main(){
	
mode = MODE_V_NT; 
//mode = MODE_V_ST; 
//mode = MODE_S_ST; 

char* source ;
char* dest ;

posix_memalign((void**)&source, 64, sizeof(char) * SIZE);
posix_memalign((void**)&dest, 64, sizeof(char) * SIZE);

//source = (char*) malloc(sizeof(char) * SIZE); 
//dest = (char*) malloc(sizeof(char) * SIZE); 
//init src array
int i = 0 ;
#pragma omp parallel for
for(i = 0 ; i < SIZE ; i++){
	source[i] = 'a' ;
	//dest[i] = 0.0 ;
}

int N ;
if(mode != MODE_S_ST){
   N = (SIZE* sizeof(double)) / (512/8);
}
else{
   N = SIZE ;
}

printf("char values : 1 => %c 25%% => %c 80%% => %c 100%% => %c \n", source[0], source[24], source[SIZE*3/4], source[SIZE-1]) ;
printf("char values dest : 1 => %c 25%% => %c 80%% => %c 100%% => %c \n", dest[0], dest[SIZE/4], dest[SIZE*3/4], dest[SIZE-1]) ;

int j = 0;
for(j = 0 ; j <50 ; j++)
	//char_copy_nontemp((uintptr_t)dest,(uintptr_t)source, SIZE);
	//char_copy3((uintptr_t)dest,(uintptr_t)source, SIZE);
	char_copy2((uintptr_t)dest,(uintptr_t)source, SIZE);
	//char_copy((uintptr_t)dest,(uintptr_t)source, SIZE);
	//simple_triad(dest, source, dest, source, N);

printf("char values : 1 => %c 25%% => %c 80%% => %c 100%% => %c \n", source[0], source[24], source[SIZE*3/4], source[SIZE-1]) ;
printf("char values dest : 1 => %c 25%% => %c 80%% => %c 100%% => %c \n", dest[0], dest[SIZE/4], dest[SIZE*3/4], dest[SIZE-1]) ;
return 1;
}

#include <sys/time.h>

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


/*
void simple_triad2(float* a , float* b, float* c, float* d, int  N);

void simple_triad2(float* a , float* b, float* c, float* d,  int N) 
{ 
int i;
//#pragma simd
#pragma vector aligned nontemporal 
 for (i=0; i<N; i++) 
 a[i] = b[i] + c[i]*d[i]; 
} 
*/


