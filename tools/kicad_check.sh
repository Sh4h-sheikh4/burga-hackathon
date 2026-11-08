#!/usr/bin/env bash
# Re-run KiCad ERC + DRC from the command line and keep the raw reports.
# Needs kicad-cli matching the project's KiCad version. Usage: tools/kicad_check.sh path/to/project_dir [out_dir]
set -euo pipefail
PROJ=${1:?project dir}; OUT=${2:-docs/raw_reports}; mkdir -p "$OUT"
# KiCad 10 project files need KiCad >= 10. Falls back to the Flathub build if no native kicad-cli exists.
if command -v kicad-cli >/dev/null; then KC=(kicad-cli)
elif flatpak info --user org.kicad.KiCad >/dev/null 2>&1 || flatpak info org.kicad.KiCad >/dev/null 2>&1; then
  KC=(flatpak run --filesystem="$(realpath "$PROJ")" --filesystem="$(realpath -m "$OUT")" --command=kicad-cli org.kicad.KiCad)
else echo "kicad-cli not found (install KiCad 10, e.g. flatpak install flathub org.kicad.KiCad)"; exit 2; fi
kicad-cli() { "${KC[@]}" "$@"; }
kicad-cli version | tee "$OUT/kicad_version.txt"
for SCH in "$PROJ"/*.kicad_sch; do
  [ -e "$SCH" ] || continue; b=$(basename "$SCH" .kicad_sch)
  kicad-cli sch erc --severity-all --format json --units mm -o "$OUT/${b}_erc.json" "$SCH" || true
  kicad-cli sch erc --severity-all --format report -o "$OUT/${b}_erc.rpt" "$SCH" || true
  kicad-cli sch export netlist --format kicadsexpr -o "$OUT/${b}.net" "$SCH" || true
done
for PCB in "$PROJ"/*.kicad_pcb; do
  [ -e "$PCB" ] || continue; b=$(basename "$PCB" .kicad_pcb)
  # --schematic-parity catches PCB/schematic mismatch, which a plain DRC run in the PCB editor can miss
  kicad-cli pcb drc --severity-all --schematic-parity --all-track-errors --refill-zones --format json --units mm -o "$OUT/${b}_drc.json" "$PCB" || true
  kicad-cli pcb drc --severity-all --schematic-parity --all-track-errors --refill-zones --format report -o "$OUT/${b}_drc.rpt" "$PCB" || true
  python3 "$(dirname "$0")/pcb_review.py" "$PCB" | tee "$OUT/${b}_pcb_review.txt" || true
done
echo "Reports in $OUT"
