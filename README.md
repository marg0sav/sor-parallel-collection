# Решение двумерного уравнения Пуассона методом SOR

Репозиторий содержит несколько параллельных реализаций метода последовательной верхней релаксации (SOR) для решения двумерного уравнения Пуассона на единичном квадрате:

$$
-\Delta u(x,y) = f(x,y), \qquad f(x,y) = \sin(\pi x)\cos(\pi y)
$$

с граничными условиями Дирихле. В качестве эталона используется аналитическое решение:

$$
u_{\text{anal}}(x,y) = -\frac{1}{2\pi^{2}}\sin(\pi x)\cos(\pi y)
$$

Во всех реализациях используется красно-чёрная (red–black) схема SOR и адаптивный выбор параметра релаксации $\omega$.

---

## Структура репозитория

- **basic_implementation/**
  - `sor_base_fixed_omega.c` — последовательный SOR с фиксированным $\omega$;
  - `sor_base_adaptive_omega.c` — последовательный SOR с адаптивным $\omega$;
  - `Makefile` — сборка и запуск базовых версий.

- **c_openmp/**
  - `sor_OpenMP_adaptive_omega.c` — SOR с красно-чёрной схемой и адаптивным $\omega$, параллелизация с помощью OpenMP;
  - `Makefile` — сборка (`make`) и запуск (`make run` с параметрами `THREADS` и `GRID`).

- **c_pthreads/**
  - `sor_pthreads_adaptive_omega.c` — SOR с использованием POSIX threads (pthreads);
  - `Makefile` — сборка (`make`) и запуск (`make run`, параметры `THREADS`, `GRID` задаются в Makefile или через командную строку).

- **c_mpi/**
  - `sor_mpi_adaptive_omega.c` — распределённая реализация SOR с использованием MPI (C);
  - `Makefile` — сборка (`mpicc`) и локальный запуск (`make run`, параметры `PROCS`, `GRID`);
  - `run_112.slurm` — пример SLURM-скрипта для запуска на кластере (112 MPI-процессов).

- **py_mpi/**
  - `sor_mpi_adaptive_omega.py` — аналогичная MPI-реализация на Python с использованием `mpi4py` и `numpy`;
  - `Makefile` — запуск через `mpirun` (`make run`, параметры `PROCS`, `GRID`);
  - `run_112.slurm` — пример SLURM-скрипта для Python/MPI-версии.

При каждом запуске программа создаёт каталог вида  
`RESULT/<имя_программы>_YYYYMMDD_HHMMSS/`  
и записывает туда файлы:
- `Analytic_solution.plt` — аналитическое решение;
- `SOR_result.plt` — численное решение SOR (Tecplot-подобный формат).

---

