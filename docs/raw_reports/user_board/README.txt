Raw KiCad 10.0.8 kicad-cli outputs for the user's Hackathon_Fall26 project (run on a scratch copy; originals untouched).
  erc.rpt / erc.json             ERC, first run with no symbol/footprint library tables configured (adds lib warnings)
  erc_libs.rpt / erc_libs.json   ERC with the stock KiCad 10 libraries configured  <- use this one
  drc.rpt / drc.json             DRC --schematic-parity --all-track-errors (no library tables)
  drc_refilled.rpt               same + --refill-zones
  drc_libs.rpt / drc_libs.json   DRC --schematic-parity --all-track-errors --refill-zones with stock libraries <- use this
  Hackathon_Fall26.net           netlist exported from the schematic
  pcb_review.txt / .json         tools/pcb_review.py datasheet checks on the .kicad_pcb
  kicad_version.txt              kicad-cli version
Command lines: see tools/kicad_check.sh (kicad-cli via flatpak org.kicad.KiCad 10.0.8).
