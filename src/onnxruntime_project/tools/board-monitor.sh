#!/bin/sh
# Board-side monitor, deliberately low-overhead: it must not become the interference it measures.
#
#   board-monitor.sh <seconds> <interval> <out.csv> [tag]
#
# Per sample it makes one `spacemit-tcm-smi` call, reads six small /proc and /sys files, and nothing
# else - no `ps -eo`, which walks all of /proc and was itself the 325 % CPU entry the first version
# reported. Interference is judged from the kernel's own pressure-stall information
# (/proc/pressure/cpu, "some avg10", one file read) and the load average. Each sample's cost is timed,
# so "low overhead" is a number the summary prints rather than an assurance.
set -eu
DURATION=${1:-60}
INTERVAL=${2:-5}
OUT=${3:-/tmp/board-monitor.csv}
TAG=${4:-run}
PSI_OK=1
[ -r /proc/pressure/cpu ] || PSI_OK=0
echo "tag,ts,tcm_busy,tcm_free,ai_khz,x100_khz,tz0,tz1,tz2,tz3,load1,psi_cpu_avg10,sample_ms" > "$OUT"
i=0
n=$((DURATION / INTERVAL))
[ "$n" -gt 0 ] || n=1
while [ "$i" -lt "$n" ]; do
    t0=$(cut -d' ' -f1 /proc/uptime)
    free=$(spacemit-tcm-smi 2>/dev/null | sed -n 's/.*available_blocks=\([0-9]*\)\/8.*/\1/p')
    ai=$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)
    x100=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)
    tz0=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
    tz1=$(cat /sys/class/thermal/thermal_zone1/temp 2>/dev/null || echo 0)
    tz2=$(cat /sys/class/thermal/thermal_zone2/temp 2>/dev/null || echo 0)
    tz3=$(cat /sys/class/thermal/thermal_zone3/temp 2>/dev/null || echo 0)
    load=$(cut -d' ' -f1 /proc/loadavg)
    psi=""
    if [ "$PSI_OK" = 1 ]; then
        psi=$(sed -n 's/^some avg10=\([0-9.]*\).*/\1/p' /proc/pressure/cpu)
    fi
    t1=$(cut -d' ' -f1 /proc/uptime)
    ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", (b - a) * 1000 }')
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$TAG" "$(date +%H:%M:%S)" \
        "$((8 - ${free:-8}))" "${free:-8}" "$ai" "$x100" "$tz0" "$tz1" "$tz2" "$tz3" \
        "$load" "$psi" "$ms" >> "$OUT"
    sleep "$INTERVAL"
    i=$((i + 1))
done
python3 - "$OUT" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no samples"); raise SystemExit
num = lambda n: [float(r.get(n) or 0) for r in rows]
temps = [max(num(f"tz{i}")) / 1000 for i in range(4)]
psi = num("psi_cpu_avg10"); cost = num("sample_ms")
exc = [r["ts"] for r in rows if float(r.get("psi_cpu_avg10") or 0) > 5]
print(f"samples={len(rows)} tcm_free_min={min(num('tcm_free')):.0f} tcm_busy_max={max(num('tcm_busy')):.0f} "
      f"tz_max_c={max(temps):.1f} load1_max={max(num('load1')):.2f} psi_cpu_max={max(psi + [0]):.1f} "
      f"sample_cost_mean_ms={sum(cost) / len(cost):.1f}")
print("EXCURSION: CPU pressure above 5 % at " + ", ".join(exc[:3]) if exc
      else "clean: CPU pressure stayed below 5 %")
PYEOF
