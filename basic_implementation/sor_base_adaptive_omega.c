#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// ----------------------------
// Константы и параметры сетки
// ----------------------------
#define ROW   51        // число узлов по x
#define COL   51        // число узлов по y
#define PI    3.141592
#define ITMAX 100000    // максимум итераций

// ----------------------------
// Прототипы
// ----------------------------
void poisson_solver(double **u, double **u_anal, double tol, double omega,
                    int BC, const char *dir_name);

void initialization(double **p);
void write_u(const char *dir_nm, const char *file_nm,
             double **p, double dx, double dy);

// SOR
void SOR(double **p, double dx, double dy, double tol, double omega,
         double *tot_time, int *iter, int BC);

// Математические функции
double func(int i, int j, double dx, double dy);
void func_anal(double **p, int row_num, int col_num, double dx, double dy);
void error_rms(double **p, double **p_anal, double *err);

// ----------------------------
// main
// ----------------------------
int main(int argc, char **argv)
{
    double **u      = NULL;
    double **u_anal = NULL;

    char dir_name[512];      // <- теперь это буфер, а не константа
    char cmd[600];
    char prog_name[256];
    char timestamp[64];

    int i;
    int BC;
    double tol, omega;

    // -------------------------------
    // Формируем имя подпапки:
    // ./RESULT/<имя_файла>_YYYYMMDD_HHMMSS/
    // -------------------------------

    // Базовая папка
    const char *base_dir = "./RESULT";

    // Имя программы: берём basename от argv[0]
    const char *raw_name = (argc > 0 && argv[0]) ? argv[0] : "run";
    const char *slash = strrchr(raw_name, '/');
#ifdef _WIN32
    const char *bslash = strrchr(raw_name, '\\');
    if (bslash && (!slash || bslash > slash)) slash = bslash;
#endif
    snprintf(prog_name, sizeof(prog_name), "%s", slash ? slash + 1 : raw_name);

    // Дата/время
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

    // Итоговый путь: ./RESULT/progname_YYYYMMDD_HHMMSS/
    snprintf(dir_name, sizeof(dir_name), "%s/%s_%s/", base_dir, prog_name, timestamp);

    // создаём базовую папку RESULT
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", base_dir);
    (void)system(cmd);

    // создаём подпапку с именем файла и датой
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir_name);
    (void)system(cmd);

    // -------------------------------
    // Выделение памяти
    // -------------------------------
    u      = (double **)malloc(ROW * sizeof(double *));
    u_anal = (double **)malloc(ROW * sizeof(double *));
    if (!u || !u_anal) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (i = 0; i < ROW; ++i) {
        u[i]      = (double *)malloc(COL * sizeof(double));
        u_anal[i] = (double *)malloc(COL * sizeof(double));
        if (!u[i] || !u_anal[i]) {
            fprintf(stderr, "Allocation failed\n");
            return 1;
        }
    }

    // --------------------
    // Начальные настройки
    // --------------------
    tol   = 1e-6;
    omega = 1.8;
    BC    = 2;   // 1 или 2

    printf("\n");
    printf("---------------------------------------- \n");
    printf("Output dir : %s\n", dir_name);
    printf("Nx : %d, Ny : %d\n", ROW, COL);
    printf("Tolerance : %e, Omega (initial) : %f \n", tol, omega);
    printf("BC = %d, method = SOR\n", BC);
    printf("---------------------------------------- \n\n");

    poisson_solver(u, u_anal, tol, omega, BC, dir_name);

    // Освобождение памяти
    for (i = 0; i < ROW; ++i) {
        free(u[i]);
        free(u_anal[i]);
    }
    free(u);
    free(u_anal);

    return 0;
}


// ----------------------------
// Правая часть f(x,y)
// ----------------------------
double func(int i, int j, double dx, double dy)
{
    double x = i * dx;
    double y = j * dy;
    return sin(PI * x) * cos(PI * y);
}

// ----------------------------
// Аналитическое решение
// u^a(x,y) = -1/(2π²) * sin(πx) cos(πy)
// ----------------------------
void func_anal(double **p, int row_num, int col_num, double dx, double dy)
{
    int i, j;
    for (i = 0; i < row_num; ++i) {
        for (j = 0; j < col_num; ++j) {
            double x = i * dx;
            double y = j * dy;
            p[i][j] = -1.0 / (2.0 * PI * PI) * sin(PI * x) * cos(PI * y);
        }
    }
}

// ----------------------------
// Инициализация (нулевое начальное приближение)
// ----------------------------
void initialization(double **p)
{
    int i, j;
    for (i = 0; i < ROW; ++i) {
        for (j = 0; j < COL; ++j) {
            p[i][j] = 0.0;
        }
    }
}

// ----------------------------
// RMS ошибка: ||u - u_anal||_2 / (N*M)
// ----------------------------
void error_rms(double **p, double **p_anal, double *err)
{
    int i, j;
    double sum = 0.0;
    for (i = 0; i < ROW; ++i) {
        for (j = 0; j < COL; ++j) {
            double diff = p[i][j] - p_anal[i][j];
            sum += diff * diff;
        }
    }
    *err = sqrt(sum) / (ROW * COL);
}

// ----------------------------
// Запись результата в файл (Tecplot-like формат)
// ----------------------------
void write_u(const char *dir_nm, const char *file_nm,
             double **p, double dx, double dy)
{
    FILE *stream;
    int i, j;
    char file_path[256];

    snprintf(file_path, sizeof(file_path), "%s%s", dir_nm, file_nm);
    stream = fopen(file_path, "w");
    if (!stream) {
        fprintf(stderr, "Cannot open file %s for writing\n", file_path);
        return;
    }

    fprintf(stream, "ZONE I=%d J=%d\n", ROW, COL);
    for (i = 0; i < ROW; ++i) {
        for (j = 0; j < COL; ++j) {
            double x = i * dx;
            double y = j * dy;
            fprintf(stream, "%f %f %f\n", x, y, p[i][j]);
        }
    }
    fclose(stream);
}

// ----------------------------
// Обёртка-солвер
// ----------------------------
void poisson_solver(double **u, double **u_anal, double tol, double omega,
                    int BC, const char *dir_name)
{
    const char *file_name = NULL;

    int iter = 0;
    double Lx = 1.0, Ly = 1.0;
    double dx, dy, err = 0.0, tot_time = 0.0;

    dx = Lx / (ROW - 1);
    dy = Ly / (COL - 1);

    // 1) аналитическое решение для сравнения
    file_name = "Analytic_solution.plt";
    func_anal(u_anal, ROW, COL, dx, dy);
    write_u(dir_name, file_name, u_anal, dx, dy);

    // 2) SOR Method 
    initialization(u);
    SOR(u, dx, dy, tol, omega, &tot_time, &iter, BC);
    error_rms(u, u_anal, &err);
    printf("SOR Method - Error : %e, Iteration : %d, Time : %f s\n",
           err, iter, tot_time);

    file_name = "SOR_result.plt";
    write_u(dir_name, file_name, u, dx, dy);
}


// ----------------------------
// SOR 
// ----------------------------
void SOR(double **p, double dx, double dy, double tol, double omega_in,
         double *tot_time, int *iter, int BC)
{
    int i, j, it;
    double beta = dx / dy;
    double SUM1, SUM2;
    double **p_new;

    clock_t start_t = clock();

    // выделяем временный массив
    p_new = (double **)malloc(ROW * sizeof(double *));
    for (i = 0; i < ROW; ++i) {
        p_new[i] = (double *)malloc(COL * sizeof(double));
    }
    initialization(p_new);

    // -----------------------------
    // NEW: оценка спектрального радиуса Jacobi и начальное omega
    // -----------------------------
    int nx = ROW - 2;                 // внутренние узлы по x
    int ny = COL - 2;                 // внутренние узлы по y
    int m  = (nx > ny) ? nx : ny;     // "характерный" размер сетки

    double rhoj   = 1.0 - PI * PI * 0.5 / ((m + 2.0) * (m + 2.0));
    double rhojsq = rhoj * rhoj;

    // локальная переменная для текущего omega
    double omega = 1.0;               // стартовое значение

    // если пользователь передал что-то осмысленное – можно взять как старт
    if (omega_in > 0.0 && omega_in < 2.0) {
        omega = omega_in;
    }
    // -----------------------------

    for (it = 1; it <= ITMAX; ++it) {
        SUM1 = 0.0;
        SUM2 = 0.0;

        // -----------------------------
        // Рекуррентное обновление omega
        // -----------------------------
        if (it == 1) {
            // аналог первого шага: 1 / (1 - 0.5 * rhojsq)
            omega = 1.0 / (1.0 - 0.5 * rhojsq);
        } else {
            // дальше: omega_{k+1} = 1 / (1 - 0.25 * rhojsq * omega_k)
            omega = 1.0 / (1.0 - 0.25 * rhojsq * omega);
        }
        // -----------------------------

        // внутренняя область
        for (i = 1; i < ROW - 1; ++i) {
            for (j = 1; j < COL - 1; ++j) {
                // сначала "чистый" GS
                double rhs = -dx * dx * func(i, j, dx, dy);
                double u_gs =
                    ( p[i+1][j] + p_new[i-1][j]
                    + beta*beta * (p[i][j+1] + p_new[i][j-1])
                    + rhs ) / (2.0 * (1.0 + beta*beta));

                // затем SOR-релаксация с текущим omega
                p_new[i][j] = p[i][j] + omega * (u_gs - p[i][j]);
            }
        }

        // ------------------------
        //  Граничные условия
        // ------------------------
        if (BC == 1) {
            int jj;
            for (jj = 0; jj < COL; ++jj) {
                p_new[0][jj]     = 0.0;
                p_new[ROW-1][jj] = 0.0;
            }
            for (i = 0; i < ROW; ++i) {
                p_new[i][0]      = p_new[i][1];
                p_new[i][COL-1]  = p_new[i][COL-2];
            }
        } else if (BC == 2) {
            int jj;
            for (jj = 0; jj < COL; ++jj) {
                p_new[0][jj]      = -1.0/(2.0*PI*PI) * func(0,       jj, dx, dy);
                p_new[ROW-1][jj]  = -1.0/(2.0*PI*PI) * func(ROW-1,  jj, dx, dy);
            }
            for (i = 0; i < ROW; ++i) {
                p_new[i][0]      = -1.0/(2.0*PI*PI) * func(i,    0, dx, dy);
                p_new[i][COL-1]  = -1.0/(2.0*PI*PI) * func(i, COL-1, dx, dy);
            }
        }

        // ------------------------
        //  Критерий сходимости
        // ------------------------
        for (i = 1; i < ROW - 1; ++i) {
            for (j = 1; j < COL - 1; ++j) {
                double val = p_new[i][j];
                SUM1 += fabs(val);
                double res = p_new[i+1][j] + p_new[i-1][j]
                           + beta*beta * (p_new[i][j+1] + p_new[i][j-1])
                           - (2.0 + 2.0*beta*beta)*val
                           - dx*dx * func(i, j, dx, dy);
                SUM2 += fabs(res);
            }
        }

        if (SUM1 > 0.0 && SUM2 / SUM1 < tol) {
            *iter = it;
            break;
        }

        // обновление p ← p_new
        for (i = 0; i < ROW; ++i) {
            for (j = 0; j < COL; ++j) {
                p[i][j] = p_new[i][j];
            }
        }
    }

    clock_t end_t = clock();
    *tot_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;

    for (i = 0; i < ROW; ++i) {
        free(p_new[i]);
    }
    free(p_new);
}
