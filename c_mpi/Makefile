CC      = mpicc
CFLAGS  = -O2 -lm

TARGET  = sor_mpi_adaptive_omega
SRC     = sor_mpi_adaptive_omega.c

PROCS  ?= 4        # число MPI процессов
GRID   ?= 51       # размер сетки

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS)

run: $(TARGET)
	mpirun -np $(PROCS) ./$(TARGET) $(GRID)

clean:
	rm -f $(TARGET) *.o
