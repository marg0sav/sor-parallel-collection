PYTHON = python3
SCRIPT = sor_mpi_adaptive_omega.py

PROCS ?= 4
GRID  ?= 51

run:
	mpirun -np $(PROCS) $(PYTHON) $(SCRIPT) $(GRID)

clean:
	rm -f *.pyc __pycache__ -r
