#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>


#define mpi_return(ret_code) MPI_Finalize(); return ret_code
#define mpi_exit(exit_code) MPI_Abort(MPI_COMM_WORLD, exit_code); \
                            exit(exit_code)

#define forn(i, n) for (int i = 0; i < n; ++i)
#define fore(i, a, b) for (int i = a; i < b; ++i)


#define _DEBUG 1


#if _DEBUG
#define debug(args...) printf(args)
#else
#define debug(args...)
#endif


#define debug0(args...) if (PROC_ID == 0) debug(args)
#define log(str, args...) debug("(%d/%d) " str "\n", PROC_ID + 1, N_PROC, args)


double PI = acos(-1);
int N_PROC, PROC_ID;

const int ROW_NUM = 100;


void shape(int t, int *from, int *to) {
    t -= ROW_NUM / 2;
    int l = (int)std::max(1.0, sqrt(pow(ROW_NUM / 2.0, 2) - pow(t, 2)));
    l *= 2;
    *from = -l;
    *to = l;
}


float border(int x, int y) {
    return sqrt(x * x + y * y);
}


struct TProcess {
private:
    typedef float TCell;

    int rowNum;
    int blockFrom, blockTo;

    std::vector<int> shift;
    std::vector<int> length;
    std::vector<int> blockPrefixLength;

    TCell *data[2];
    int curStepDataPtrId;

    void calcCurrentBlock() {
        int curBlockSize = (rowNum - 2) / N_PROC;
        debug0("curBlockSize = %d\n", curBlockSize);
        if (PROC_ID < (rowNum - 2) % N_PROC) curBlockSize++;

        blockFrom = (rowNum - 2) / N_PROC * PROC_ID +
            std::min((int)(rowNum - 2) % N_PROC, PROC_ID);
        blockTo = blockFrom + curBlockSize + 1;
    }

    TCell* allocDataArray(int dataSize) {
        TCell *data = (TCell*)malloc(dataSize * sizeof(TCell));

        if (data == NULL) {
            printf("%d can't allocate %d bytes\n",
                   PROC_ID + 1, dataSize * sizeof(TCell));
            mpi_exit(1);
        }

        memset(data, 0, dataSize * sizeof(TCell));
        log("Successfully allocated %d bytes",
            dataSize * sizeof(TCell));

        return data;
    }

    inline int idx(int curRow, int curShift) {
        int id = curShift - shift[curRow];
        curRow -= blockFrom;
        id += blockPrefixLength[curRow];
        return id;
    }


    bool isBorder(int curRow, int curShift) {
        if (curShift == shift[curRow] ||
                curShift == shift[curRow] + length[curRow] - 1)
        {
            return true;
        }

        if (curRow == 0 || curRow == rowNum - 1) {
            return true;
        }

        return (curShift < shift[curRow - 1] ||
                curShift >= shift[curRow - 1] + length[curRow - 1] ||
                curShift < shift[curRow + 1] ||
                curShift >= shift[curRow + 1] + length[curRow + 1]);
    }

    void initDataArray(float (*border)(int, int)) {
        fore(r, blockFrom, blockTo + 1) {
            fore(c, shift[r], shift[r] + length[r]) {
                if (isBorder(r, c)) {
                    data[1][idx(r, c)] = data[0][idx(r, c)] = border(r, c);
                }
            }
        }
    }

    TCell get(int curRow, int curShift) {
        return data[curStepDataPtrId][idx(curRow, curShift)];
    }

    void set(int curRow, int curShift, TCell val) {
        data[curStepDataPtrId ^ 1][idx(curRow, curShift)] = val;
    }

    void doIter() {
        fore(curRow, blockFrom + 1, blockTo) {
            int l = shift[curRow];
            int r = l + length[curRow];

            // intersect with previous row
            l = std::max(l, shift[curRow - 1]);
            r = std::min(r, shift[curRow - 1] + (int)length[curRow - 1]);

            // intersect with next row
            l = std::max(l, shift[curRow + 1]);
            r = std::min(r, shift[curRow + 1] + (int)length[curRow + 1]);

            if (isBorder(curRow, l)) l++;
            if (isBorder(curRow, r)) r--;

            fore(curShift, l, r + 1) {
                TCell val = (
                    get(curRow - 1, curShift) + get(curRow + 1, curShift) +
                    get(curRow, curShift - 1) + get(curRow, curShift + 1));
                set(curRow, curShift, val / 4.0);
            }
        }

        curStepDataPtrId ^= 1;
    }

    void writeRows(FILE *fs, int from, int to) {
        fore(r, from, to + 1) {
            for (int s = shift[r]; s < shift[r] + length[r]; ++s) {
                fprintf(fs, "%d %d %.5f\n", r, s, get(r, s));
            }
        }
    }

public:
    TProcess(int rowNum, void (*shape)(int, int*, int*), float (*border)(int, int)) {
         curStepDataPtrId = 0;

        this->rowNum = rowNum;
        shift.resize(rowNum);
        length.resize(rowNum);

        forn(i, rowNum) {
            int from, to;
            shape(i, &from, &to);
            shift[i] = from;
            length[i] = to - from + 1;
        }

        calcCurrentBlock();
        log("[%d .. %d]", blockFrom, blockTo);
        blockPrefixLength.resize(blockTo - blockFrom + 2);
        fore(i, blockFrom, blockTo + 1) {
            int id = i - blockFrom;
            log("Row %d starts with index %d", i, blockPrefixLength[id]);
            blockPrefixLength[id + 1] = blockPrefixLength[id] + length[i];
        }
        log("Totally has %d elements in data array",
            blockPrefixLength[blockTo - blockFrom + 1]);

        forn(i, 2) {
            data[i] = allocDataArray(blockPrefixLength.back());
        }

        initDataArray(border);
    }

    void run(int maxIterNum, double eps) {
        forn(iter, maxIterNum) {
            MPI_Barrier(MPI_COMM_WORLD);
            doIter();
        }
    }

    void writeToFile(const char* output) {
        FILE *fs;

        if (PROC_ID == 0) {
            fs = fopen(output, "w");
            fclose(fs);
        }

        for (int i = 0; i < N_PROC; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (PROC_ID == i) {
                fs = fopen(output, "a");

                if (PROC_ID == 0) {
                    writeRows(fs, 0, 0);
                }

                writeRows(fs, blockFrom + 1, blockTo - 1);

                if (PROC_ID == N_PROC - 1) {
                    writeRows(fs, blockTo, blockTo);
                }

                fclose(fs);
            }
        }
    }

    ~TProcess() {
        forn(i, 2) {
            free(data[i]);
        }
    }
};


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &N_PROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &PROC_ID);

    TProcess proc(ROW_NUM, shape, border);
    proc.run(2000, 1e-5);
    proc.writeToFile("data.txt");

    mpi_return(0);
}
