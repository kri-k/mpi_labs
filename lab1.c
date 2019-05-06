#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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


const char* foutName = "output_1.bin";

int N_PROC, PROC_ID;

size_t MAX_ITER;
size_t M, N, W; // i, j, k
size_t FROM, TO;


inline void init(void) {
    MPI_Comm_size(MPI_COMM_WORLD, &N_PROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &PROC_ID); 

    MAX_ITER = 1000;

    // stack W matrices of sizes (M, N)
    M = 50;
    N = 50;
    W = 50;
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

    MPI_Request sendReq;
    MPI_Status status;

    MPI_Request recvReqs[2];
    int waitRecv[2] = {};

    for (size_t iter = 1; iter <= MAX_ITER; ++iter) {
        // Asynchronously receive data
        if (PROC_ID > 0) {
            waitRecv[0] = True;
            MPI_Irecv(data[1], M * N, MPI_FLOAT, PROC_ID - 1,
                      0, MPI_COMM_WORLD, &recvReqs[0]);
        } else {
            waitRecv[0] = False;
        }
        if (PROC_ID < N_PROC - 1) {
            waitRecv[1] = True;
            MPI_Irecv(data[1] + IDX(0, 0, TO), M * N, MPI_FLOAT, PROC_ID + 1,
                      1, MPI_COMM_WORLD, &recvReqs[1]);
        } else {
            waitRecv[1] = False;
        }

        // Calc next iteration
        float *d = data[0];
        for (size_t w = FROM + 1; w <= TO - 1; ++w) {
            for (size_t i = 1; i < M - 1; ++i) {
                for (size_t j = 1; j < N - 1; ++j) {
                    data[1][IDX(i, j, w)] = (
                        d[IDX(i + 1, j, w)] + d[IDX(i - 1, j, w)] +
                        d[IDX(i, j + 1, w)] + d[IDX(i, j - 1, w)] +
                        d[IDX(i, j, w + 1)] + d[IDX(i, j, w - 1)]
                    ) / 6.0;
                }
            }
        }

        // Send data
        if (PROC_ID > 0) {
            MPI_Isend(data[1] + IDX(0, 0, FROM + 1), M * N, MPI_FLOAT, PROC_ID - 1,
                      1, MPI_COMM_WORLD, &sendReq);
            MPI_Wait(&sendReq, &status);
        }
        if (PROC_ID < N_PROC - 1) {
            MPI_Isend(data[1] + IDX(0, 0, TO - 1), M * N, MPI_FLOAT, PROC_ID + 1,
                      0, MPI_COMM_WORLD, &sendReq);
            MPI_Wait(&sendReq, &status); 
        }

        debug("%d has sent all data\n", PROC_ID + 1);

        // Wait for received data
        if (waitRecv[0]) {
            debug("%d wait for data in 0\n", PROC_ID + 1);
            MPI_Wait(&recvReqs[0], &status);
        }
        if (waitRecv[1]) {
            debug("%d wait for data in 1\n", PROC_ID + 1);
            MPI_Wait(&recvReqs[1], &status);
        }
        debug("%d has received all data\n", PROC_ID + 1);

        data[0] = data[1];
        data[1] = d;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    FILE *fs;

    if (PROC_ID == 0) {
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
    mpi_return(0);
}
