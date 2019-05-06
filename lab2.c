#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>


#define True 1
#define False 0

#define IDX(i, j, k) ((((k) - FROM) * M + (i)) * N + (j))
#define mpi_return(ret_code) MPI_Finalize(); return ret_code
#define mpi_exit(exit_code) MPI_Finalize(); exit(exit_code)


#define _DEBUG 0


#if _DEBUG
#define debug(args...) printf(args)
#else
#define debug(args...)
#endif


const char* foutName = "output_2.bin";

int N_PROC, PROC_ID;

size_t MAX_ITER;
float EPS;
size_t M, N, W; // i, j, k
size_t FROM, TO;


inline void init(void) {
    MPI_Comm_size(MPI_COMM_WORLD, &N_PROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &PROC_ID); 

    MAX_ITER = 10000;
    EPS = 0.01;

    // stack W matrices of sizes (M, N)
    M = 100;
    N = 100;
    W = 100;
}


inline int min(int a, int b) {
    return a < b ? a: b;
}


inline float borderFunc(int i, int j, int k) {
    return i + j + k;
}


void calcCurrentBlock(void) {
    int curBlockSize = (W - 2) / N_PROC;
    if (PROC_ID < (W - 2) % N_PROC) curBlockSize++;

    // Process matrices from (M, N, FROM) to (M, N, TO)
    FROM = (W - 2) / N_PROC * PROC_ID + min((W - 2) % N_PROC, PROC_ID);
    TO = FROM + curBlockSize + 1;
}


float* allocDataArray(int dataSize) {
    float *data = (float*)malloc(dataSize * sizeof(float));

    if (data == NULL) {
        printf("%d can't allocate %d bytes\n",
               PROC_ID + 1, dataSize * sizeof(float));
        mpi_exit(1);
    }

    debug("(%d/%d): Successfully allocated %d bytes\n",
           PROC_ID + 1, N_PROC, dataSize * sizeof(float));

    return data;
}


void _writeMatricesBinary(FILE *f, float* data, size_t from, size_t to) {
    debug("(%d/%d) write %d bytes\n",
          PROC_ID + 1, N_PROC, IDX(0, 0, to + 1) - IDX(0, 0, from));
    fwrite(
        data + IDX(0, 0, from), sizeof(float),
        IDX(0, 0, to + 1) - IDX(0, 0, from), f);
}


void _writeMatricesText(FILE *f, float* data, size_t from, size_t to) {
    for (int w = from; w <= to; ++w) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                fprintf(f, "%.2f ", data[IDX(i, j, w)]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "=============\n");
    }
}


void writeMatrices(FILE *f, float *data, size_t from, size_t to) {
    _writeMatricesBinary(f, data, from, to);
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    init();

    double start = MPI_Wtime();
    calcCurrentBlock();

    debug("(%d/%d): Has blocks from %d to %d\n",
           PROC_ID + 1, N_PROC, FROM, TO);
    MPI_Barrier(MPI_COMM_WORLD);

    size_t dataSize = M * N * (TO - FROM + 1);
    float* data[2] = {
        allocDataArray(dataSize),
        allocDataArray(dataSize)
    };

    // Initialize data array
    memset(data[0], 0, dataSize * sizeof(float));
    memset(data[1], 0, dataSize * sizeof(float));
    for (size_t w = FROM; w <= TO; ++w) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (w == 0 || i == 0 || j == 0 ||
                    w == W - 1 || i == M - 1 || j == N - 1)
                {
                    data[0][IDX(i, j, w)] = borderFunc(i, j, w);
                    data[1][IDX(i, j, w)] = data[0][IDX(i, j, w)];
                }
            }
        }
    }

    debug("(%d/%d): Successfully init grid\n",
           PROC_ID + 1, N_PROC);
    MPI_Barrier(MPI_COMM_WORLD);

    int *sendSizes, *shiftForSend, *recvSizes, *shiftForRecv;
    
    sendSizes = (int*)malloc(N_PROC * sizeof(int));
    shiftForSend = (int*)malloc(N_PROC * sizeof(int));

    recvSizes = (int*)malloc(N_PROC * sizeof(int));
    shiftForRecv = (int*)malloc(N_PROC * sizeof(int));

    memset(sendSizes, 0, N_PROC * sizeof(int));
    memset(shiftForSend, 0, N_PROC * sizeof(int));
    memset(recvSizes, 0, N_PROC * sizeof(int));
    memset(shiftForRecv, 0, N_PROC * sizeof(int));
    
    if (PROC_ID > 0) {
        sendSizes[PROC_ID - 1] = M * N;
        shiftForSend[PROC_ID - 1] = IDX(0, 0, FROM + 1);
        recvSizes[PROC_ID - 1] = M * N;
        shiftForRecv[PROC_ID - 1] = IDX(0, 0, FROM);
    }
    if (PROC_ID < N_PROC - 1) {
        sendSizes[PROC_ID + 1] = M * N;
        shiftForSend[PROC_ID + 1] = IDX(0, 0, TO - 1);
        recvSizes[PROC_ID + 1] = M * N;
        shiftForRecv[PROC_ID + 1] = IDX(0, 0, TO);
    }

    size_t iter;
    float localEps;
    float globalEps;
    for (iter = 1; iter <= MAX_ITER; ++iter) {
        // Calc next iteration
        localEps = 0;
        float *d = data[0];
        for (size_t w = FROM + 1; w <= TO - 1; ++w) {
            for (size_t i = 1; i < M - 1; ++i) {
                for (size_t j = 1; j < N - 1; ++j) {
                    data[1][IDX(i, j, w)] = (
                        d[IDX(i + 1, j, w)] + d[IDX(i - 1, j, w)] +
                        d[IDX(i, j + 1, w)] + d[IDX(i, j - 1, w)] +
                        d[IDX(i, j, w + 1)] + d[IDX(i, j, w - 1)]
                    ) / 6.0;
                    localEps = fmax(
                        localEps, fabs(data[1][IDX(i, j, w)] - d[IDX(i, j, w)]));
                }
            }
        }

        MPI_Alltoallv(
            data[1], sendSizes, shiftForSend, MPI_FLOAT,
            data[1], recvSizes, shiftForRecv, MPI_FLOAT,
            MPI_COMM_WORLD);
        debug("%d has sent/receive all data\n", PROC_ID + 1);

        data[0] = data[1];
        data[1] = d;
        
        MPI_Allreduce(&localEps, &globalEps, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        if (PROC_ID == 0) {
            debug("EPS: %f\n", globalEps);
        }
        if (globalEps <= EPS) {
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    FILE *fs;

    if (PROC_ID == 0) {
        printf("Iter num: %d\n", iter);
        printf("EPS: %f\n", globalEps);
        fs = fopen(foutName, "w");
        fclose(fs);
    }

    float *d = data[0];
    for (int i = 0; i < N_PROC; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (PROC_ID == i) {
            fs = fopen(foutName, "ab");

            if (PROC_ID == 0) {
                writeMatrices(fs, d, 0, 0);
            }

            writeMatrices(fs, d, FROM + 1, TO - 1);

            if (PROC_ID == N_PROC - 1) {
                writeMatrices(fs, d, TO, TO);
            }

            fclose(fs);
        }
    }

    printf("%lf\n", MPI_Wtime() - start);
    free(data[0]);
    free(data[1]);
    
    free(sendSizes);
    free(shiftForSend);
    free(recvSizes);
    free(shiftForRecv);

    mpi_return(0);
}
