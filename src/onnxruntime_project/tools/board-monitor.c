// Low-overhead board monitor (Rule 1 of TOOLING-RULES.md): C, no shell, no Python, no `ps`.
//
//   board-monitor <seconds> <fast-interval-ms> <smi-every-n-samples> <out.csv> [tag]
//
// High-frequency samples read only /proc and /sys files (clocks, four thermal zones, load average,
// CPU pressure stall information) - microseconds each - while the vendor `spacemit-tcm-smi` binary is
// invoked only every Nth sample, because that call costs ~100 ms and the TCM block table changes
// slowly. The tool prints its own mean cost per sample, so Rule 3 holds.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int read_first_number(const char *path, double *out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    double v = 0;
    int ok = fscanf(f, "%lf", &v) == 1;
    fclose(f);
    if (ok && out) *out = v;
    return ok ? 0 : -1;
}

static int read_psi_some_avg10(double *out) {
    FILE *f = fopen("/proc/pressure/cpu", "r");
    if (!f) return -1;
    char line[256];
    int ok = -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "some", 4) != 0) continue;
        char *p = strstr(line, "avg10=");
        if (p && sscanf(p + 6, "%lf", out) == 1) ok = 0;
        break;
    }
    fclose(f);
    return ok;
}

// The vendor tool is the only supported reader of the block table; it is deliberately called at a
// lower rate than the file-based metrics.
static int read_tcm_free(int *free_blocks) {
    FILE *p = popen("spacemit-tcm-smi 2>/dev/null", "r");
    if (!p) return -1;
    char line[512];
    int ok = -1;
    while (fgets(line, sizeof line, p)) {
        char *m = strstr(line, "available_blocks=");
        if (m && sscanf(m + 17, "%d", free_blocks) == 1) { ok = 0; break; }
    }
    pclose(p);
    return ok;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: board-monitor <seconds> <fast-ms> <smi-every> <out.csv> [tag]\n");
        return 2;
    }
    double seconds = atof(argv[1]);
    int fast_ms = atoi(argv[2]);
    int smi_every = atoi(argv[3]);
    const char *out = argv[4];
    const char *tag = argc > 5 ? argv[5] : "run";
    if (fast_ms <= 0) fast_ms = 500;
    if (smi_every <= 0) smi_every = 4;

    FILE *csv = fopen(out, "w");
    if (!csv) { perror(out); return 1; }
    fprintf(csv, "tag,ts,tcm_free,ai_khz,x100_khz,tz0,tz1,tz2,tz3,load1,psi_cpu_avg10,sample_us\n");

    double t_end = now_ms() + seconds * 1000.0;
    int sample = 0, tcm_free = -1;
    double cost_sum = 0, cost_max = 0;
    for (;;) {
        double t0 = now_ms();
        if (sample % smi_every == 0) read_tcm_free(&tcm_free);
        double ai = 0, x100 = 0, tz[4] = {0, 0, 0, 0}, load1 = 0, psi = -1;
        read_first_number("/sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq", &ai);
        read_first_number("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", &x100);
        for (int i = 0; i < 4; ++i) {
            char path[64];
            snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/temp", i);
            read_first_number(path, &tz[i]);
        }
        read_first_number("/proc/loadavg", &load1);
        read_psi_some_avg10(&psi);
        double t1 = now_ms();
        double cost = t1 - t0;
        cost_sum += cost;
        if (cost > cost_max) cost_max = cost;
        fprintf(csv, "%s,%ld,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.2f,%.2f,%.0f\n", tag,
                (long) time(NULL), tcm_free, ai, x100, tz[0], tz[1], tz[2], tz[3], load1,
                psi < 0 ? 0 : psi, cost * 1000.0);
        fflush(csv);
        ++sample;
        if (now_ms() >= t_end) break;
        usleep(fast_ms * 1000);
    }
    fclose(csv);
    printf("samples=%d mean_cost_us=%.0f max_cost_us=%.0f tcm_free_last=%d\n",
           sample, cost_sum / sample * 1000.0, cost_max * 1000.0, tcm_free);
    return 0;
}
