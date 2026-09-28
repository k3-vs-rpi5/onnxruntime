// One-sample process memory inspector (Rule 1 of TOOLING-RULES.md): C, file reads only, ~microseconds.
//
//   proc-mem-inspect <pid> <model-file-substring> [tag]
//
// Reads /proc/<pid>/stat (minflt, majflt) and /proc/<pid>/smaps, and reports:
//   * RSS and the huge-page accounting the kernel exposes (AnonHugePages, FilePmdMapped),
//   * how much of the *model file* mapping is backed by huge pages, which is the mechanism question
//     behind the mmap penalty,
//   * the number of distinct file mappings of that model and their total size.
// One line per invocation, cheap enough to run inside a measurement window.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: proc-mem-inspect <pid> <model-file-substring> [tag]\n");
        return 2;
    }
    const char *pid = argv[1];
    const char *needle = argv[2];
    const char *tag = argc > 3 ? argv[3] : "-";

    char path[256];
    long minflt = 0, majflt = 0, rss_kb = 0, anon_huge_kb = 0;
    snprintf(path, sizeof path, "/proc/%s/stat", pid);
    FILE *f = fopen(path, "r");
    if (f) {
        char line[4096];
        if (fgets(line, sizeof line, f)) {
            // fields: pid (comm) state ppid ... minflt(10) ... majflt(12); comm may contain spaces, so
            // parse from the closing parenthesis.
            char *p = strrchr(line, ')');
            if (p) {
                long v[40];
                int n = 0;
                for (char *tok = strtok(p + 2, " "); tok && n < 40; tok = strtok(NULL, " "))
                    v[n++] = atol(tok);
                // after ')' the sequence starts at field 3; minflt is field 10 -> index 7, majflt 12 -> 9
                if (n > 9) { minflt = v[7]; majflt = v[9]; }
            }
        }
        fclose(f);
    }

    long file_pmd_kb = 0, model_kb = 0, model_pmd_kb = 0;
    int model_maps = 0;
    int in_model = 0;
    long cur_size = 0, cur_pmd = 0, cur_rss = 0;
    snprintf(path, sizeof path, "/proc/%s/smaps", pid);
    FILE *s = fopen(path, "r");
    if (s) {
        char line[1024];
        while (fgets(line, sizeof line, s)) {
            if (strchr(line, '-') && strstr(line, " r") && strstr(line, " 00000000 ")) {
                // a new mapping header: settle the previous one
                if (in_model) {
                    model_maps++;
                    model_kb += cur_size;
                    model_pmd_kb += cur_pmd;
                }
                in_model = strstr(line, needle) != NULL;
                cur_size = cur_pmd = cur_rss = 0;
                continue;
            }
            if (sscanf(line, "Size: %ld kB", &cur_size) == 1) continue;
            if (sscanf(line, "Rss: %ld kB", &cur_rss) == 1) { rss_kb += cur_rss; continue; }
            if (sscanf(line, "AnonHugePages: %ld kB", &anon_huge_kb) == 1) continue;
            if (sscanf(line, "FilePmdMapped: %ld kB", &cur_pmd) == 1) {
                file_pmd_kb += cur_pmd;
                continue;
            }
            if (sscanf(line, "ShmemPmdMapped: %ld kB", &cur_pmd) == 1) continue;
        }
        if (in_model) {
            model_maps++;
            model_kb += cur_size;
            model_pmd_kb += cur_pmd;
        }
        fclose(s);
    }
    printf("%s minflt=%ld majflt=%ld rss_mb=%.1f anon_huge_mb=%.1f file_pmd_mb=%.1f "
           "model_maps=%d model_mb=%.1f model_huge_mb=%.1f\n",
           tag, minflt, majflt, rss_kb / 1024.0, anon_huge_kb / 1024.0, file_pmd_kb / 1024.0,
           model_maps, model_kb / 1024.0, model_pmd_kb / 1024.0);
    return 0;
}
