CC      = gcc
CFLAGS  = -O2 -std=gnu99 -pthread -lm

TARGET  = sor_pthreads_adaptive_omega
SRC     = sor_pthreads_adaptive_omega.c

SIZE    ?= 51   # размер сетки Nx=Ny
THREADS ?= 4    # число потоков

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS)

run: $(TARGET)
	./$(TARGET) $(SIZE) $(THREADS)

clean:
	rm -f $(TARGET) *.o
