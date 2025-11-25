CC = gcc
CFLAGS = -O2 -lm

TARGETS = sor_base_fixed_omega sor_base_adaptive_omega

SOURCES = sor_base_fixed_omega.c sor_base_adaptive_omega.c

all: $(TARGETS)

sor_base_fixed_omega: sor_base_fixed_omega.c
	$(CC) $< -o $@ $(CFLAGS)

sor_base_adaptive_omega: sor_base_adaptive_omega.c
	$(CC) $< -o $@ $(CFLAGS)

run_fixed: sor_base_fixed_omega
	./sor_base_fixed_omega

run_adaptive: sor_base_adaptive_omega
	./sor_base_adaptive_omega

clean:
	rm -f $(TARGETS)
