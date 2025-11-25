#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define PI    3.141592
#define ITMAX 100000

int ROW = 0;
int COL = 0;

// Параметры разбиения по потокам (по строкам)
int NTHREADS = 1;
int *t_counts = NULL;   // сколько внутренних строк у каждого потока
int *t_displs = NULL;   // смещение (в строках) относительно глобального ig=1

// Глобальные данные для SOR
double **g_u = NULL;
double g_dx = 0.0, g_dy = 0.0, g_beta = 0.0;
double g_tol = 1e-6;
int    g_BC  = 2;
double g_rhojsq = 0.0;

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

// ----------------------------
// Разбиение внутренних строк между потоками
// interior_n = ROW - 2 (без физических границ)
// ----------------------------
void setup_decomp_threads(int nthreads)
{
    int interior = ROW - 2; // строки 1..ROW-2
    t_counts = (int*)malloc(nthreads * sizeof(int));
    t_displs = (int*)malloc(nthreads * sizeof(int));

    if (!t_counts || !t_displs) {
        fprintf(stderr, "Allocation failed (t_counts/t_displs)\n");
        exit(1);
    }

    int base = (interior > 0) ? (interior / nthreads) : 0;
    int rem  = (interior > 0) ? (interior % nthreads) : 0;

    int offset = 0;
    for (int r = 0; r < nthreads; ++r) {
        t_counts[r] = base + (r < rem ? 1 : 0);
        t_displs[r] = offset;
        offset     += t_counts[r];
    }
}

// ----------------------------
// Применение граничных условий на весь массив
// u[0 .. ROW-1][0 .. COL-1]
// BC = 1: u=0 сверху/снизу, Neumann по x слева/справа
// BC = 2: аналитическое решение на всех границах
// ----------------------------
void apply_BC(double **u, double dx, double dy, int BC)
{
    if (BC == 1) {
        // Neumann слева/справа: du/dx = 0 -> u(i,0) = u(i,1), u(i,N-1)=u(i,N-2)
        for (int i = 1; i <= ROW - 2; ++i) {
            u[i][0]      = u[i][1];
            u[i][COL-1]  = u[i][COL-2];
        }
        // Dirichlet сверху/снизу: u=0
        for (int j = 0; j < COL; ++j) {
            u[0][j]         = 0.0;
            u[ROW-1][j]     = 0.0;
        }
    } else if (BC == 2) {
        // Аналитическое на всех границах (Dirichlet)
        // Левые/правые
        for (int i = 0; i < ROW; ++i) {
            u[i][0]      = u_anal(i, 0,      dx, dy);
            u[i][COL-1]  = u_anal(i, COL-1,  dx, dy);
        }
        // Верх/низ
        for (int j = 0; j < COL; ++j) {
            u[0][j]        = u_anal(0,       j, dx, dy);
            u[ROW-1][j]    = u_anal(ROW-1,   j, dx, dy);
        }
    }
}

// ----------------------------
// Параллельный проход по цвету (red/black) через pthreads
// color = 0 -> (i+j) чётное, color = 1 -> нечётное
// ----------------------------
typedef struct {
    int  tid; // номер потока
    int  color;
    double omega;
} SweepArgs;

void* sweep_worker(void *arg)
{
    SweepArgs *a = (SweepArgs*)arg;
    int tid   = a->tid;
    int color = a->color;
    double omega = a->omega;

    int rows = t_counts[tid];
    if (rows <= 0) {
        return NULL;
    }
    int ig_start = 1 + t_displs[tid];
    int ig_end   = ig_start + rows - 1;  // включительно, в диапазоне [1..ROW-2]

    double **u   = g_u;
    double dx    = g_dx;
    double dy    = g_dy;
    double beta  = g_beta;

    for (int ig = ig_start; ig <= ig_end; ++ig) {
        for (int j = 1; j < COL - 1; ++j) {
            if ( ((ig + j) & 1) != color ) continue;

            double rhs = -dx * dx * func(ig, j, dx, dy);
            double u_gs =
                ( u[ig+1][j] + u[ig-1][j]
                + beta*beta * (u[ig][j+1] + u[ig][j-1])
                + rhs ) / (2.0 * (1.0 + beta*beta));

            u[ig][j] = u[ig][j] + omega * (u_gs - u[ig][j]);
        }
    }

    return NULL;
}

void sweep_color_parallel(int color, double omega)
{
    pthread_t *threads = (pthread_t*)malloc(NTHREADS * sizeof(pthread_t));
    SweepArgs *args    = (SweepArgs*)malloc(NTHREADS * sizeof(SweepArgs));

    if (!threads || !args) {
        fprintf(stderr, "Allocation failed in sweep_color_parallel\n");
        exit(1);
    }

    for (int t = 0; t < NTHREADS; ++t) {
        args[t].tid   = t;
        args[t].color = color;
        args[t].omega = omega;
        pthread_create(&threads[t], NULL, sweep_worker, &args[t]);
    }

    for (int t = 0; t < NTHREADS; ++t) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
}

// ----------------------------
// Параллельное вычисление невязки (SUM1,SUM2)
// SUM1 = sum |u|, SUM2 = sum |Lu - f|
// ----------------------------
typedef struct {
    int tid;
    double sum1;
    double sum2;
} ResidualArgs;

void* residual_worker(void *arg)
{
    ResidualArgs *a = (ResidualArgs*)arg;
    int tid  = a->tid;
    double sum1 = 0.0;
    double sum2 = 0.0;

    int rows = t_counts[tid];
    if (rows <= 0) {
        a->sum1 = 0.0;
        a->sum2 = 0.0;
        return NULL;
    }

    int ig_start = 1 + t_displs[tid];
    int ig_end   = ig_start + rows - 1;  // 1..ROW-2

    double **u  = g_u;
    double dx   = g_dx;
    double dy   = g_dy;
    double beta = g_beta;

    for (int ig = ig_start; ig <= ig_end; ++ig) {
        for (int j = 1; j < COL - 1; ++j) {
            double val = u[ig][j];
            sum1 += fabs(val);

            double res =
                u[ig+1][j] + u[ig-1][j]
              + beta*beta * (u[ig][j+1] + u[ig][j-1])
              - (2.0 + 2.0*beta*beta)*val
              - dx*dx * func(ig, j, dx, dy);

            sum2 += fabs(res);
        }
    }

    a->sum1 = sum1;
    a->sum2 = sum2;
    return NULL;
}

void residual_parallel(double *SUM1_glob, double *SUM2_glob)
{
    pthread_t    *threads = (pthread_t*)malloc(NTHREADS * sizeof(pthread_t));
    ResidualArgs *args    = (ResidualArgs*)malloc(NTHREADS * sizeof(ResidualArgs));

    if (!threads || !args) {
        fprintf(stderr, "Allocation failed in residual_parallel\n");
        exit(1);
    }

    for (int t = 0; t < NTHREADS; ++t) {
        args[t].tid = t;
        args[t].sum1 = 0.0;
        args[t].sum2 = 0.0;
        pthread_create(&threads[t], NULL, residual_worker, &args[t]);
    }

    double S1 = 0.0, S2 = 0.0;
    for (int t = 0; t < NTHREADS; ++t) {
        pthread_join(threads[t], NULL);
        S1 += args[t].sum1;
        S2 += args[t].sum2;
    }

    *SUM1_glob = S1;
    *SUM2_glob = S2;

    free(threads);
    free(args);
}

// ----------------------------
// SOR red-black + адаптивный w на pthreads
// ----------------------------
static inline double wall_time_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

void SOR_pthreads(double **u,
                  double dx, double dy, double tol, double omega_in,
                  double *cpu_time, double *wall_time,
                  int *iter_out, double *err_out,
                  int BC)
{
    g_u   = u;
    g_dx  = dx;
    g_dy  = dy;
    g_beta = dx / dy;
    g_tol  = tol;
    g_BC   = BC;

    int nx = ROW - 2;
    int ny = COL - 2;
    int m  = (nx > ny) ? nx : ny;

    double rhoj   = 1.0 - PI * PI * 0.5 / ((m + 2.0) * (m + 2.0));
    g_rhojsq = rhoj * rhoj;

    double omega = 1.0;
    if (omega_in > 0.0 && omega_in < 2.0) {
        omega = omega_in;
    }

    // таймеры
    double wall_start = wall_time_now();
    double cpu_start  = (double)clock() / (double)CLOCKS_PER_SEC;

    // начальные ГУ
    apply_BC(u, dx, dy, BC);

    // Предварительные два полушага
    // 1) red, ω=1
    sweep_color_parallel(0, 1.0);
    apply_BC(u, dx, dy, BC);

    // 2) black, ω = 1 / (1 - 0.5*rhojsq)
    omega = 1.0 / (1.0 - 0.5 * g_rhojsq);
    sweep_color_parallel(1, omega);
    apply_BC(u, dx, dy, BC);

    int it = 1;
    double err_glob = 0.0;

    while (it < ITMAX) {
        // --- red sweep ---
        omega = 1.0 / (1.0 - 0.25 * g_rhojsq * omega);
        sweep_color_parallel(0, omega);
        apply_BC(u, dx, dy, BC);

        // --- black sweep ---
        omega = 1.0 / (1.0 - 0.25 * g_rhojsq * omega);
        sweep_color_parallel(1, omega);
        apply_BC(u, dx, dy, BC);

        // --- критерий сходимости (по невязке) ---
        double SUM1_glob = 0.0, SUM2_glob = 0.0;
        residual_parallel(&SUM1_glob, &SUM2_glob);

        if (SUM1_glob > 0.0)
            err_glob = SUM2_glob / SUM1_glob;
        else
            err_glob = SUM2_glob;

        if (err_glob < tol)
            break;

        ++it;
    }

    // оценка ошибки относительно аналитического решения (RMS) — последним шагом, последовательно
    double err_sq = 0.0;
    long long count = 0;
    for (int ig = 0; ig < ROW; ++ig) {
        for (int j = 0; j < COL; ++j) {
            double u_exact = u_anal(ig, j, dx, dy);
            double diff = u[ig][j] - u_exact;
            err_sq += diff * diff;
            ++count;
        }
    }
    double rms_err = sqrt(err_sq) / (double)count;

    double wall_end = wall_time_now();
    double cpu_end  = (double)clock() / (double)CLOCKS_PER_SEC;

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
    if (argc >= 2) {
        ROW = atoi(argv[1]);
    } else {
        ROW = 51;
    }
    COL = ROW;

    if (argc >= 3) {
        NTHREADS = atoi(argv[2]);
    } else {
        // попытаться взять число физических ядер, иначе 4
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        NTHREADS = (n > 0) ? (int)n : 4;
    }
    if (NTHREADS < 1) NTHREADS = 1;

    // NEW: подготовка директории вывода (как в MPI-версии)
    char dir_name[512] = "";
    {
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

    // Параметры задачи
    double tol   = 1e-6;
    double omega = 1.0;   // если не (0,2), будет пересчитан адаптивно
    int    BC    = 2;

    double Lx = 1.0, Ly = 1.0;
    double dx = Lx / (ROW - 1);
    double dy = Ly / (COL - 1);

    // Разбиение по потокам (по внутренним строкам 1..ROW-2)
    setup_decomp_threads(NTHREADS);

    // Аллокация глобального массива u[0..ROW-1][0..COL-1]
    double **u = (double**)malloc(ROW * sizeof(double*));
    if (!u) {
        fprintf(stderr, "Allocation failed (u pointers)\n");
        exit(1);
    }
    u[0] = (double*)malloc(ROW * COL * sizeof(double));
    if (!u[0]) {
        fprintf(stderr, "Allocation failed (u data)\n");
        exit(1);
    }
    for (int i = 1; i < ROW; ++i) {
        u[i] = u[0] + i * COL;
    }

    // Инициализация
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            u[i][j] = 0.0;
        }
    }

    printf("\n----------------------------------------\n");
    printf("Pthreads SOR red-black\n");
    printf("Output dir : %s\n", dir_name);
    printf("Threads : %d\n", NTHREADS);
    printf("Nx : %d, Ny : %d\n", ROW, COL);
    printf("Tolerance : %e, Omega(initial) : %f\n", tol, omega);
    printf("BC = %d\n", BC);
    printf("----------------------------------------\n\n");

    double cpu_time, wall_time, err;
    int iter;

    SOR_pthreads(u, dx, dy, tol, omega,
                 &cpu_time, &wall_time, &iter, &err,
                 BC);

    printf("SOR Pthreads - RMS Error : %e, Iteration : %d\n", err, iter);
    printf("  CPU  (approx) : %f s\n", cpu_time);
    printf("  Wall time     : %f s\n", wall_time);

    // Запись файлов: аналитика + SOR
    double *u_flat = &u[0][0];
    double *u_anal_glob = (double*)malloc(ROW * COL * sizeof(double));
    if (!u_anal_glob) {
        fprintf(stderr, "Allocation failed (u_anal_glob)\n");
        exit(1);
    }

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            u_anal_glob[i * COL + j] = u_anal(i, j, dx, dy);
        }
    }

    write_u_serial(dir_name, "Analytic_solution.plt", u_anal_glob, dx, dy);
    write_u_serial(dir_name, "SOR_result.plt",        u_flat,      dx, dy);

    free(u_anal_glob);
    free(u[0]);
    free(u);
    free(t_counts);
    free(t_displs);

    return 0;
}
