CC      = gcc
CFLAGS  = -O2 -fopenmp -lm

TARGET  = sor_OpenMP_adaptive_omega
SRC     = sor_OpenMP_adaptive_omega.c

THREADS ?= 4

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS)

run: $(TARGET)
	OMP_NUM_THREADS=$(THREADS) ./$(TARGET) 51

clean:
	rm -f $(TARGET) *.o
