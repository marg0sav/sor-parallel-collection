#!/usr/bin/env python3
import sys
import os
import time
import math
import numpy as np
from mpi4py import MPI

PI = 3.141592
ITMAX = 100000

ROW = 0
COL = 0

# ----------------------------
# f(x,y)
# ----------------------------
def func(ig, jg, dx, dy):
    x = ig * dx
    y = jg * dy
    return math.sin(PI * x) * math.cos(PI * y)


# аналитическое решение
def u_anal(ig, jg, dx, dy):
    x = ig * dx
    y = jg * dy
    return -1.0 / (2.0 * PI * PI) * math.sin(PI * x) * math.cos(PI * y)


# Запись Tecplot-подобного файла по глобальному массиву (1D, ROW*COL)
def write_u_serial(dir_nm, file_nm, p, dx, dy):
    file_path = os.path.join(dir_nm, file_nm)
    try:
        with open(file_path, "w") as f:
            f.write(f"ZONE I={ROW} J={COL}\n")
            for i in range(ROW):
                for j in range(COL):
                    x = i * dx
                    y = j * dy
                    f.write(f"{x:.15e} {y:.15e} {p[i * COL + j]:.15e}\n")
    except OSError:
        sys.stderr.write(f"Cannot open file {file_path} for writing\n")


# Сбор решения со всех процессов и запись файлов
def output_results_MPI(u, local_n, i_start, dx, dy, dir_name, counts, displs, comm, rank, size):
    global ROW, COL

    interior = ROW - 2  # количество внутренних строк по i

    # Буфер отправки: только внутренние строки [1..local_n]
    sendbuf = u[1:local_n+1, :].ravel()  # 1D

    recvcounts = None
    displs_gv = None
    buf_int = None
    u_glob = None
    u_anal_glob = None

    if rank == 0:
        recvcounts = np.array(counts, dtype=np.int32) * COL
        displs_gv = np.array(displs, dtype=np.int32) * COL
        buf_int = np.empty(interior * COL, dtype=np.float64)

    # Собираем только внутренние строки (1..ROW-2)
    comm.Gatherv(sendbuf,
                 [buf_int, recvcounts, displs_gv, MPI.DOUBLE] if rank == 0 else None,
                 root=0)

    if rank == 0:
        # Глобальные массивы: численное и аналитическое решение
        u_glob = np.empty(ROW * COL, dtype=np.float64)
        u_anal_glob = np.empty(ROW * COL, dtype=np.float64)

        # Заполняем аналитическое решение полностью
        for i in range(ROW):
            for j in range(COL):
                u_anal_glob[i * COL + j] = u_anal(i, j, dx, dy)

        # Численное решение: внутренние строки берём из buf_int,
        # граничные строки (0 и ROW-1) заполняем аналитическим решением
        for j in range(COL):
            u_glob[0 * COL + j] = u_anal(0, j, dx, dy)
            u_glob[(ROW - 1) * COL + j] = u_anal(ROW - 1, j, dx, dy)

        # Внутренние строки: ig = 1..ROW-2
        buf_int_2d = buf_int.reshape((interior, COL))
        for ig in range(1, ROW - 1):
            u_glob[ig * COL:(ig + 1) * COL] = buf_int_2d[ig - 1, :]

        # Пишем два файла: аналитика и результат SOR
        write_u_serial(dir_name, "Analytic_solution.plt", u_anal_glob, dx, dy)
        write_u_serial(dir_name, "SOR_result.plt", u_glob, dx, dy)


# ----------------------------
# Разбиение строк между процессами
# interior_n = ROW - 2 (без физических границ)
# ----------------------------
def setup_decomp(size, rank, counts, displs):
    global ROW
    interior = ROW - 2  # строки 1..ROW-2
    base = interior // size
    rem = interior % size

    offset = 0
    for r in range(size):
        cnt = base + (1 if r < rem else 0)
        counts[r] = cnt
        displs[r] = offset
        offset += cnt

    local_n = counts[rank]
    i_start = 1 + displs[rank]  # глобальный индекс первой внутренней строки
    return local_n, i_start


# ----------------------------
# Применение граничных условий на локальный блок
# u[0 .. local_n+1][0 .. COL-1]
# ----------------------------
def apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size):
    global ROW, COL

    ig_top = i_start               # глобальный индекс первой внутренней строки
    ig_bottom = i_start + local_n - 1  # глобальный индекс последней внутренней

    # Физические границы по j (левая/правая) присутствуют у всех процессов
    if BC == 1:
        # u=0 сверху/снизу, Neumann по x слева/справа
        for il in range(1, local_n + 1):
            u[il, 0] = u[il, 1]
            u[il, COL - 1] = u[il, COL - 2]
    elif BC == 2:
        for il in range(1, local_n + 1):
            ig = i_start + (il - 1)
            u[il, 0] = u_anal(ig, 0, dx, dy)
            u[il, COL - 1] = u_anal(ig, COL - 1, dx, dy)

    # Физические границы по i (сверху/снизу) — только у крайних процессов.
    # Эти значения запишем в halo-строки u[0][j] и u[local_n+1][j],
    # если процесс "касается" соответствующей границы.
    if BC == 1:
        if ig_top == 1:  # над ней физическая граница ig=0, u=0
            u[0, :] = 0.0
        if ig_bottom == ROW - 2:  # под ней физическая граница ig=ROW-1, u=0
            u[local_n + 1, :] = 0.0
    elif BC == 2:
        if ig_top == 1:  # физическая граница ig=0
            for j in range(COL):
                u[0, j] = u_anal(0, j, dx, dy)
        if ig_bottom == ROW - 2:  # физическая граница ig=ROW-1
            for j in range(COL):
                u[local_n + 1, j] = u_anal(ROW - 1, j, dx, dy)


# ----------------------------
# Обмен halo-строк
# u[0] <-> верхний сосед,
# u[local_n+1] <-> нижний сосед.
# ----------------------------
def exchange_halo(u, local_n, rank, size, comm):
    global COL

    up = MPI.PROC_NULL if rank == 0 else rank - 1
    down = MPI.PROC_NULL if rank == size - 1 else rank + 1

    # Отправляем свою первую внутреннюю строку вверх, получаем нижнюю halo
    comm.Sendrecv(sendbuf=u[1, :], dest=up, sendtag=0,
                  recvbuf=u[local_n + 1, :], source=down, recvtag=0)

    # Отправляем свою последнюю внутреннюю строку вниз, получаем верхнюю halo
    comm.Sendrecv(sendbuf=u[local_n, :], dest=down, sendtag=1,
                  recvbuf=u[0, :], source=up, recvtag=1)


# ----------------------------
# Один проход по цвету (red/black)
# color = 0 -> (i+j) чётное, color = 1 -> нечётное
# ----------------------------
def sweep_color(u, local_n, i_start, dx, dy, beta, color, omega_local):
    global COL
    for il in range(1, local_n + 1):
        ig = i_start + (il - 1)  # глобальный i
        for j in range(1, COL - 1):
            if ((ig + j) & 1) != color:
                continue
            rhs = -dx * dx * func(ig, j, dx, dy)
            u_gs = (u[il + 1, j] + u[il - 1, j] +
                    beta * beta * (u[il, j + 1] + u[il, j - 1]) +
                    rhs) / (2.0 * (1.0 + beta * beta))
            u[il, j] = u[il, j] + omega_local * (u_gs - u[il, j])


# ----------------------------
# Основной SOR red-black + адаптивный w (MPI)
# ----------------------------
def SOR_MPI(u, local_n, i_start, dx, dy, tol, omega_in,
            BC, comm, rank, size):

    global ROW, COL, ITMAX

    beta = dx / dy

    wall_start = MPI.Wtime()
    cpu_start = MPI.Wtime()  # как в C: "cpu" ~ wall

    # оценка спектрального радиуса Jacobi
    nx = ROW - 2
    ny = COL - 2
    m = nx if nx > ny else ny
    rhoj = 1.0 - PI * PI * 0.5 / ((m + 2.0) * (m + 2.0))
    rhojsq = rhoj * rhoj

    omega = 1.0
    if 0.0 < omega_in < 2.0:
        omega = omega_in

    # начальные ГУ + halo
    apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size)
    exchange_halo(u, local_n, rank, size, comm)

    # Предварительные два полушага
    # 1) red, ω=1
    sweep_color(u, local_n, i_start, dx, dy, beta, 0, 1.0)
    apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size)
    exchange_halo(u, local_n, rank, size, comm)

    # 2) black, ω = 1 / (1 - 0.5*rhojsq)
    omega = 1.0 / (1.0 - 0.5 * rhojsq)
    sweep_color(u, local_n, i_start, dx, dy, beta, 1, omega)
    apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size)
    exchange_halo(u, local_n, rank, size, comm)

    it = 1
    err_glob = 0.0

    while it < ITMAX:
        # --- red sweep ---
        omega = 1.0 / (1.0 - 0.25 * rhojsq * omega)
        sweep_color(u, local_n, i_start, dx, dy, beta, 0, omega)

        # обновляем физические границы и рассылаем halo
        apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size)
        exchange_halo(u, local_n, rank, size, comm)

        # --- black sweep ---
        omega = 1.0 / (1.0 - 0.25 * rhojsq * omega)
        sweep_color(u, local_n, i_start, dx, dy, beta, 1, omega)

        apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size)
        exchange_halo(u, local_n, rank, size, comm)

        # --- критерий сходимости ---
        SUM1_loc = 0.0
        SUM2_loc = 0.0
        for il in range(1, local_n + 1):
            ig = i_start + (il - 1)
            for j in range(1, COL - 1):
                val = u[il, j]
                SUM1_loc += abs(val)
                res = (u[il + 1, j] + u[il - 1, j] +
                       beta * beta * (u[il, j + 1] + u[il, j - 1]) -
                       (2.0 + 2.0 * beta * beta) * val -
                       dx * dx * func(ig, j, dx, dy))
                SUM2_loc += abs(res)

        SUM1_glob = comm.allreduce(SUM1_loc, op=MPI.SUM)
        SUM2_glob = comm.allreduce(SUM2_loc, op=MPI.SUM)

        if SUM1_glob > 0.0:
            err_glob = SUM2_glob / SUM1_glob
        else:
            err_glob = SUM2_glob

        if err_glob < tol:
            break

        it += 1

    # оценка ошибки относительно аналитического решения (RMS-подобная метрика)
    err_sq_loc = 0.0
    count_loc = 0

    for il in range(0, local_n + 2):
        ig = i_start + (il - 1)  # для il=0 -> ig=i_start-1
        for j in range(COL):
            if ig < 0 or ig > ROW - 1:
                continue
            u_exact = u_anal(ig, j, dx, dy)
            diff = u[il, j] - u_exact
            err_sq_loc += diff * diff
            count_loc += 1

    err_sq_glob = comm.allreduce(err_sq_loc, op=MPI.SUM)
    count_glob = comm.allreduce(count_loc, op=MPI.SUM)

    rms_err = math.sqrt(err_sq_glob) / float(count_glob)

    wall_end = MPI.Wtime()
    cpu_end = MPI.Wtime()

    cpu_time = cpu_end - cpu_start
    wall_time = wall_end - wall_start

    return rms_err, it, cpu_time, wall_time


# ----------------------------
# main
# ----------------------------
def main():
    global ROW, COL

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Чтение размера сетки на rank 0
    if rank == 0:
        if len(sys.argv) >= 2:
            ROW = int(sys.argv[1])
        else:
            ROW = 51
        COL = ROW

    # Рассылаем размеры сетки
    ROW = comm.bcast(ROW if rank == 0 else None, root=0)
    COL = ROW

    # Подготовка директории вывода (на корне)
    if rank == 0:
        base_dir = "./RESULT"
        raw_name = sys.argv[0] if len(sys.argv) > 0 else "run"
        prog_name = os.path.basename(raw_name)

        now = time.localtime()
        timestamp = time.strftime("%Y%m%d_%H%M%S", now)

        dir_name = os.path.join(base_dir, f"{prog_name}_{timestamp}")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(dir_name, exist_ok=True)
    else:
        dir_name = None

    # Разослать имя директории всем
    dir_name = comm.bcast(dir_name, root=0)

    # Параметры задачи
    tol = 1e-6
    omega = 1.0  # если не (0,2), будет пересчитан адаптивно
    BC = 2
    Lx, Ly = 1.0, 1.0
    dx = Lx / (ROW - 1)
    dy = Ly / (COL - 1)

    # Разбиение по процессам
    counts = np.zeros(size, dtype=np.int32)
    displs = np.zeros(size, dtype=np.int32)
    local_n, i_start = setup_decomp(size, rank, counts, displs)

    if local_n <= 0:
        if rank == 0:
            sys.stderr.write("Too many MPI processes for this grid size.\n")
        MPI.Finalize()
        return

    # Аллокация локального массива u[0..local_n+1][0..COL-1]
    u = np.zeros((local_n + 2, COL), dtype=np.float64)

    if rank == 0:
        print("\n----------------------------------------")
        print("MPI SOR red-black (Python + mpi4py)")
        print(f"Output dir : {dir_name}")
        print(f"Procs      : {size}")
        print(f"Nx, Ny     : {ROW}, {COL}")
        print(f"Tolerance  : {tol:e}, Omega(initial) : {omega:.6f}")
        print(f"BC         : {BC}")
        print("----------------------------------------\n")

    # Запуск SOR
    rms_err, it, cpu_time, wall_time = SOR_MPI(
        u, local_n, i_start, dx, dy, tol, omega, BC, comm, rank, size
    )

    # Вывод информации и файлов (только печать на rank 0)
    if rank == 0:
        print(f"SOR MPI - RMS Error : {rms_err:e}, Iteration : {it}")
        print(f" CPU (approx) : {cpu_time:.6f} s")
        print(f" Wall time    : {wall_time:.6f} s")

    # Сбор и запись полей
    output_results_MPI(u, local_n, i_start, dx, dy, dir_name, counts, displs, comm, rank, size)

    MPI.Finalize()


if __name__ == "__main__":
    main()
