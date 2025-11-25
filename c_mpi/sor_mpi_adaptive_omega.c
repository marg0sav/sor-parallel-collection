#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#define PI    3.141592
#define ITMAX 100000

int ROW = 0;
int COL = 0;

// ----------------------------
// f(x,y)
// ----------------------------
static inline double func(int ig, int jg, double dx, double dy)
{
    double x = ig * dx;
    double y = jg * dy;
    return sin(PI * x) * cos(PI * y);
}

// аналитическое решение
static inline double u_anal(double ig, double jg, double dx, double dy)
{
    double x = ig * dx;
    double y = jg * dy;
    return -1.0 / (2.0 * PI * PI) * sin(PI * x) * cos(PI * y);
}

// Запись Tecplot-подобного файла по глобальному массиву (1D, ROW*COL)
void write_u_serial(const char *dir_nm, const char *file_nm,
                    double *p, double dx, double dy)
{
    FILE *stream;
    char file_path[512];

    snprintf(file_path, sizeof(file_path), "%s%s", dir_nm, file_nm);
    stream = fopen(file_path, "w");
    if (!stream) {
        fprintf(stderr, "Cannot open file %s for writing\n", file_path);
        return;
    }

    fprintf(stream, "ZONE I=%d J=%d\n", ROW, COL);
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            double x = i * dx;
            double y = j * dy;
            fprintf(stream, "%f %f %f\n", x, y, p[i * COL + j]);
        }
    }
    fclose(stream);
}

// Сбор решения со всех процессов и запись файлов
void output_results_MPI(double **u,
                        int local_n, int i_start,
                        double dx, double dy,
                        const char *dir_name,
                        int *counts, int *displs,
                        MPI_Comm comm, int rank, int size)
{
    int interior = ROW - 2;  // количество внутренних строк по i

    // Буфер отправки: только внутренние строки [1..local_n]
    double *sendbuf = &u[1][0];
    int sendcount = local_n * COL;

    int *recvcounts = NULL;
    int *displs_gv  = NULL;
    double *buf_int = NULL;
    double *u_glob = NULL;
    double *u_anal_glob = NULL;

    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs_gv  = (int *)malloc(size * sizeof(int));
        if (!recvcounts || !displs_gv) {
            fprintf(stderr, "Allocation failed (recvcounts/displs_gv)\n");
            MPI_Abort(comm, 1);
        }

        for (int r = 0; r < size; ++r) {
            recvcounts[r] = counts[r] * COL;   // счётчик в элементах
            displs_gv[r]  = displs[r] * COL;   // смещение в элементах
        }

        buf_int = (double *)malloc(interior * COL * sizeof(double));
        if (!buf_int) {
            fprintf(stderr, "Allocation failed (buf_int)\n");
            MPI_Abort(comm, 1);
        }
    }

    // Собираем только внутренние строки (1..ROW-2)
    MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE,
                buf_int, recvcounts, displs_gv, MPI_DOUBLE,
                0, comm);

    if (rank == 0) {
        // Глобальные массивы: численное и аналитическое решение
        u_glob      = (double *)malloc(ROW * COL * sizeof(double));
        u_anal_glob = (double *)malloc(ROW * COL * sizeof(double));
        if (!u_glob || !u_anal_glob) {
            fprintf(stderr, "Allocation failed (u_glob/u_anal_glob)\n");
            MPI_Abort(comm, 1);
        }

        // Заполняем аналитическое решение полностью
        for (int i = 0; i < ROW; ++i) {
            for (int j = 0; j < COL; ++j) {
                u_anal_glob[i * COL + j] = u_anal(i, j, dx, dy);
            }
        }

        // Численное решение: внутренние строки берём из buf_int,
        // граничные строки (0 и ROW-1) заполняем аналитическим решением
        for (int j = 0; j < COL; ++j) {
            u_glob[0 * COL + j]        = u_anal(0, j, dx, dy);
            u_glob[(ROW - 1) * COL + j] = u_anal(ROW - 1, j, dx, dy);
        }

        for (int ig = 1; ig <= ROW - 2; ++ig) {
            for (int j = 0; j < COL; ++j) {
                u_glob[ig * COL + j] = buf_int[(ig - 1) * COL + j];
            }
        }

        // Пишем два файла: аналитика и результат SOR
        write_u_serial(dir_name, "Analytic_solution.plt", u_anal_glob, dx, dy);
        write_u_serial(dir_name, "SOR_result.plt",        u_glob,      dx, dy);

        free(recvcounts);
        free(displs_gv);
        free(buf_int);
        free(u_glob);
        free(u_anal_glob);
    }
}

// ----------------------------
// Разбиение строк между процессами
// interior_n = ROW - 2 (без физических границ)
// ----------------------------
void setup_decomp(int size, int rank, int *local_n, int *i_start,
                  int *counts, int *displs)
{
    int interior = ROW - 2; // строки 1..ROW-2
    int base = interior / size;
    int rem  = interior % size;

    for (int r = 0, offset = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0); //сколько внутренних строк получает процесс r
        displs[r] = offset;
        offset   += counts[r];
    }

    *local_n = counts[rank]; // сколько строк (по i) внутри этого процесса призрачных
    *i_start = 1 + displs[rank]; // глобальный индекс первой внутренней строки
}

// ----------------------------
// Применение граничных условий на локальный блок
// u[0 .. local_n+1][0 .. COL-1]
// ----------------------------
void apply_BC_local(double **u, int local_n, int i_start,
                    double dx, double dy, int BC,
                    int rank, int size)
{
    int ig_top    = i_start;            // глобальный индекс первой внутренней строки
    int ig_bottom = i_start + local_n - 1; // глобальный индекс последней внутренней

    // Физические ГРАНИЦЫ по j (левая/правая) присутствуют у всех процессов
    if (BC == 1) {
        // u=0 сверху/снизу, Neumann по x слева/справа
        for (int il = 1; il <= local_n; ++il) {
            u[il][0]      = u[il][1];
            u[il][COL-1]  = u[il][COL-2];
        }
    } else if (BC == 2) {
        for (int il = 1; il <= local_n; ++il) {
            int ig = i_start + (il - 1);
            u[il][0]      = u_anal(ig,    0, dx, dy);
            u[il][COL-1]  = u_anal(ig, COL-1, dx, dy);
        }
    }

    // Физические ГРАНИЦЫ по i (сверху/снизу) — только у крайних процессов.
    // Эти значения запишем в halo-строки u[0][j] и u[local_n+1][j],
    // если процесс "касается" соответствующей границы.
    if (BC == 1) {
        if (ig_top == 1) {
            // над ней физическая граница ig = 0, u=0
            for (int j = 0; j < COL; ++j) {
                u[0][j] = 0.0;
            }
        }
        if (ig_bottom == ROW-2) {
            // под ней физическая граница ig = ROW-1, u=0
            for (int j = 0; j < COL; ++j) {
                u[local_n+1][j] = 0.0;
            }
        }
    } else if (BC == 2) {
        if (ig_top == 1) {
            // физическая граница ig = 0
            for (int j = 0; j < COL; ++j) {
                u[0][j] = u_anal(0, j, dx, dy);
            }
        }
        if (ig_bottom == ROW-2) {
            // физическая граница ig = ROW-1
            for (int j = 0; j < COL; ++j) {
                u[local_n+1][j] = u_anal(ROW-1, j, dx, dy);
            }
        }
    }
}

// ----------------------------
// Обмен halo-строк
// u[0] <-> верхний сосед,
// u[local_n+1] <-> нижний сосед.
// ----------------------------
void exchange_halo(double **u, int local_n, int rank, int size, MPI_Comm comm)
{
    // определяем соседей
    int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Отправляем свою первую внутреннюю строку вверх, получаем нижнюю halo
    MPI_Sendrecv(u[1],           COL, MPI_DOUBLE, up,   0,
                 u[local_n+1],   COL, MPI_DOUBLE, down, 0,
                 comm, MPI_STATUS_IGNORE);

    // Отправляем свою последнюю внутреннюю строку вниз, получаем верхнюю halo
    MPI_Sendrecv(u[local_n],     COL, MPI_DOUBLE, down, 1,
                 u[0],           COL, MPI_DOUBLE, up,   1,
                 comm, MPI_STATUS_IGNORE);
}

// ----------------------------
// Один проход по цвету (red/black)
// color = 0 -> (i+j) чётное, color = 1 -> нечётное
// ----------------------------
void sweep_color(double **u, int local_n, int i_start,
                 double dx, double dy, double beta,
                 int color, double omega_local)
{
    for (int il = 1; il <= local_n; ++il) {
        int ig = i_start + (il - 1);  // глобальный i
        for (int j = 1; j < COL - 1; ++j) {
            if ( ((ig + j) & 1) != color ) continue;

            double rhs = -dx * dx * func(ig, j, dx, dy);
            double u_gs =
                ( u[il+1][j] + u[il-1][j]
                + beta*beta * (u[il][j+1] + u[il][j-1])
                + rhs ) / (2.0 * (1.0 + beta*beta));

            u[il][j] = u[il][j] + omega_local * (u_gs - u[il][j]);
        }
    }
}

// ----------------------------
// Основной SOR red-black + адаптивный w (MPI)
// ----------------------------
void SOR_MPI(double **u,
             int local_n, int i_start,
             double dx, double dy, double tol, double omega_in,
             double *cpu_time, double *wall_time,
             int *iter_out, double *err_out,
             int BC, MPI_Comm comm, int rank, int size)
{
    double beta = dx / dy;
    double SUM1_loc, SUM2_loc;
    double err_loc, err_glob;

    // таймеры
    double wall_start = MPI_Wtime();
    double cpu_start  = MPI_Wtime(); // грубая оценка "cpu" через wall

    // оценка спектрального радиуса Jacobi
    int nx = ROW - 2;
    int ny = COL - 2;
    int m  = (nx > ny) ? nx : ny;

    double rhoj   = 1.0 - PI * PI * 0.5 / ((m + 2.0) * (m + 2.0));
    double rhojsq = rhoj * rhoj;

    double omega = 1.0;
    if (omega_in > 0.0 && omega_in < 2.0) {
        omega = omega_in;
    }

    // начальные ГУ + halo
    apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size);
    exchange_halo(u, local_n, rank, size, comm);

    // Предварительные два полушага
    // 1) red, ω=1
    sweep_color(u, local_n, i_start, dx, dy, beta, 0, 1.0);
    apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size);
    exchange_halo(u, local_n, rank, size, comm);

    // 2) black, ω = 1 / (1 - 0.5*rhojsq)
    omega = 1.0 / (1.0 - 0.5 * rhojsq);
    sweep_color(u, local_n, i_start, dx, dy, beta, 1, omega);
    apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size);
    exchange_halo(u, local_n, rank, size, comm);

    int it = 1;

    while (it < ITMAX) {
    // --- red sweep ---
    omega = 1.0 / (1.0 - 0.25 * rhojsq * omega);
    sweep_color(u, local_n, i_start, dx, dy, beta, 0, omega);

    // обновляем физические границы и рассылаем halo,
    // чтобы соседние процессы видели новые "красные" значения
    apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size);
    exchange_halo(u, local_n, rank, size, comm);

    // --- black sweep ---
    omega = 1.0 / (1.0 - 0.25 * rhojsq * omega);
    sweep_color(u, local_n, i_start, dx, dy, beta, 1, omega);

    // снова физические границы + обмен halo,
    // теперь уже с обновлёнными "чёрными" значениями
    apply_BC_local(u, local_n, i_start, dx, dy, BC, rank, size);
    exchange_halo(u, local_n, rank, size, comm);

    // --- критерий сходимости ---
    SUM1_loc = 0.0;
    SUM2_loc = 0.0;

    for (int il = 1; il <= local_n; ++il) {
        int ig = i_start + (il - 1);
        for (int j = 1; j < COL - 1; ++j) {
            double val = u[il][j];
            SUM1_loc += fabs(val);

            double res =
                u[il+1][j] + u[il-1][j]
              + beta*beta * (u[il][j+1] + u[il][j-1])
              - (2.0 + 2.0*beta*beta)*val
              - dx*dx * func(ig, j, dx, dy);

            SUM2_loc += fabs(res);
        }
    }

    double SUM1_glob = 0.0, SUM2_glob = 0.0;
    MPI_Allreduce(&SUM1_loc, &SUM1_glob, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&SUM2_loc, &SUM2_glob, 1, MPI_DOUBLE, MPI_SUM, comm);

    if (SUM1_glob > 0.0)
        err_glob = SUM2_glob / SUM1_glob;
    else
        err_glob = SUM2_glob;

    if (err_glob < tol)
        break;

    ++it;
}


    // оценка ошибки относительно аналитического решения (RMS)
    double err_sq_loc = 0.0;
    long long count_loc = 0;

    for (int il = 0; il <= local_n+1; ++il) {
        int ig = i_start + (il - 1); // для il=0 -> ig=i_start-1 (верхняя граница или сосед)
        for (int j = 0; j < COL; ++j) {
            // учитываем только реальный глобальный диапазон 0..ROW-1
            if (ig < 0 || ig > ROW-1) continue;

            double u_exact = u_anal(ig, j, dx, dy);
            double diff = u[il][j] - u_exact;
            err_sq_loc += diff * diff;
            ++count_loc;
        }
    }

    double err_sq_glob = 0.0;
    long long count_glob = 0;
    MPI_Allreduce(&err_sq_loc, &err_sq_glob, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&count_loc,  &count_glob,  1, MPI_LONG_LONG, MPI_SUM, comm);

    double rms_err = sqrt(err_sq_glob) / (double)count_glob;

    double wall_end = MPI_Wtime();
    double cpu_end  = MPI_Wtime();

    *cpu_time  = cpu_end  - cpu_start;
    *wall_time = wall_end - wall_start;
    *iter_out  = it;
    *err_out   = rms_err;
}

// ----------------------------
// main
// ----------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        if (argc >= 2) {
            ROW = atoi(argv[1]);
        } else {
            ROW = 51;
        }
        COL = ROW;
    }

    // Рассылаем размеры сетки
    MPI_Bcast(&ROW, 1, MPI_INT, 0, comm);
    COL = ROW;

    // NEW: подготовка директории вывода
    char dir_name[512] = "";
    if (rank == 0) {
        const char *base_dir = "./RESULT";
        char cmd[600];
        char prog_name[256];
        char timestamp[64];

        const char *raw_name = (argc > 0 && argv[0]) ? argv[0] : "run";
        const char *slash = strrchr(raw_name, '/');
#ifdef _WIN32
        const char *bslash = strrchr(raw_name, '\\');
        if (bslash && (!slash || bslash > slash)) slash = bslash;
#endif
        snprintf(prog_name, sizeof(prog_name), "%s", slash ? slash + 1 : raw_name);

        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

        // ./RESULT/progname_YYYYMMDD_HHMMSS/
        snprintf(dir_name, sizeof(dir_name), "%s/%s_%s/", base_dir, prog_name, timestamp);

        // создаём базовую папку RESULT
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", base_dir);
        (void)system(cmd);

        // создаём подпапку
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir_name);
        (void)system(cmd);
    }

    // Разослать имя директории всем (на всякий случай)
    MPI_Bcast(dir_name, 512, MPI_CHAR, 0, comm);

    // Параметры задачи
    double tol   = 1e-6;
    double omega = 1.0;   // если не (0,2), будет пересчитан адаптивно
    int    BC    = 2;

    double Lx = 1.0, Ly = 1.0;
    double dx = Lx / (ROW - 1);
    double dy = Ly / (COL - 1);

    // Разбиение по процессам
    int *counts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    int local_n, i_start;
    setup_decomp(size, rank, &local_n, &i_start, counts, displs);

    if (local_n <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Too many MPI processes for this grid size.\n");
        }
        free(counts);
        free(displs);
        MPI_Finalize();
        return 1;
    }

    // Аллокация локального массива u[0..local_n+1][0..COL-1]
    double **u = (double**)malloc((local_n + 2) * sizeof(double*));
    u[0] = (double*)malloc((local_n + 2) * COL * sizeof(double));
    for (int i = 1; i < local_n + 2; ++i) {
        u[i] = u[0] + i * COL;
    }

    // Инициализация
    for (int il = 0; il <= local_n+1; ++il) {
        for (int j = 0; j < COL; ++j) {
            u[il][j] = 0.0;
        }
    }

    if (rank == 0) {
        printf("\n----------------------------------------\n");
        printf("MPI SOR red-black\n");
        printf("Output dir : %s\n", dir_name);   // NEW
        printf("Procs : %d\n", size);
        printf("Nx : %d, Ny : %d\n", ROW, COL);
        printf("Tolerance : %e, Omega(initial) : %f\n", tol, omega);
        printf("BC = %d\n", BC);
        printf("----------------------------------------\n\n");
    }

    double cpu_time, wall_time, err;
    int iter;

    SOR_MPI(u, local_n, i_start, dx, dy, tol, omega,
            &cpu_time, &wall_time, &iter, &err,
            BC, comm, rank, size);

    // Вывод информации и файлов
    if (rank == 0) {
        printf("SOR MPI - RMS Error : %e, Iteration : %d\n", err, iter);
        printf("  CPU  (approx) : %f s\n", cpu_time);
        printf("  Wall time     : %f s\n", wall_time);
    }

    // NEW: собираем поле и пишем Analytic_solution.plt и SOR_result.plt
    output_results_MPI(u, local_n, i_start, dx, dy,
                       dir_name, counts, displs,
                       comm, rank, size);

    free(u[0]);
    free(u);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}