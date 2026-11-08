#!/usr/bin/env python3
"""Electrical sanity review of a KiCad .kicad_pcb (KiCad 6/7/8/9 s-expression; KiCad 5 'module' also parsed).

It reads pad->net assignments (which reflect the schematic after 'Update PCB from Schematic') and checks
the Raspberry Pi 40-pin header and the LSM6DS3TR-C wiring against the datasheet. It does NOT replace
KiCad ERC/DRC (run tools/kicad_check.sh) and cannot judge physical things (solder, reflow, vibration).
    python tools/pcb_review.py board.kicad_pcb [--json]
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([()])|([^\s()"]+)')


def parse_sexpr(text: str):
    stack, cur = [], []
    for m in TOKEN.finditer(text):
        s, paren, atom = m.groups()
        if paren == "(":
            stack.append(cur)
            cur = []
        elif paren == ")":
            done, cur = cur, stack.pop()
            cur.append(done)
        else:
            cur.append(s if s is not None else atom)
    return cur[0] if cur else []


def children(node, name):
    return [c for c in node if isinstance(c, list) and c and c[0] == name]


def first(node, name):
    c = children(node, name)
    return c[0] if c else None


# LSM6DS3TR-C LGA-14, DocID030071 Table 2
IMU_PINS = {"1": "SDO/SA0", "2": "SDx", "3": "SCx", "4": "INT1", "5": "VDDIO", "6": "GND", "7": "GND",
            "8": "VDD", "9": "INT2", "10": "NC", "11": "NC", "12": "CS", "13": "SCL", "14": "SDA"}
PI_3V3, PI_5V = ("1", "17"), ("2", "4")
PI_GND = ("6", "9", "14", "20", "25", "30", "34", "39")
PI_SDA1, PI_SCL1, PI_ID = "3", "5", ("27", "28")
PI_SPI0 = {"19": "MOSI", "21": "MISO", "23": "SCLK", "24": "CE0", "26": "CE1"}


def load_board(text: str) -> dict:
    root = parse_sexpr(text)
    fps = []
    for fp in children(root, "footprint") + children(root, "module"):
        lib = fp[1] if len(fp) > 1 and isinstance(fp[1], str) else ""
        ref = val = ""
        for p in children(fp, "property"):
            if p[1] == "Reference":
                ref = p[2]
            elif p[1] == "Value":
                val = p[2]
        for t in children(fp, "fp_text"):
            if t[1] == "reference":
                ref = ref or t[2]
            elif t[1] == "value":
                val = val or t[2]
        pads, pos = {}, {}
        at = first(fp, "at") or ["at", "0", "0"]
        x0, y0 = float(at[1]), float(at[2])
        rot = math.radians(-(float(at[3]) if len(at) > 3 else 0.0))
        for pad in children(fp, "pad"):
            num = pad[1]
            net = first(pad, "net")
            # KiCad <=9: (net 3 "name"); KiCad 10 may use (net "name")
            name = None if net is None else (net[-1] if len(net) >= 2 else None)
            if num in ("", None):
                continue
            pads.setdefault(num, name)  # first pad wins for multi-pad numbers
            pa = first(pad, "at")
            if pa:
                px, py = float(pa[1]), float(pa[2])
                pos.setdefault(num, (x0 + px * math.cos(rot) - py * math.sin(rot),
                                     y0 + px * math.sin(rot) + py * math.cos(rot)))
        fps.append({"ref": ref, "value": val, "lib": lib, "pads": pads, "pos": pos})
    widths = defaultdict(list)
    netnames = {n[1]: n[2] for n in children(root, "net") if len(n) >= 3}
    for seg in children(root, "segment"):
        w, net = first(seg, "width"), first(seg, "net")
        if w and net:
            name = netnames.get(net[1], net[1])
            widths[name].append(float(w[1]))
    setup = first(root, "setup")
    stackup = first(setup, "stackup") if setup else None
    layers = [l[1] for l in (first(root, "layers") or [])[1:] if isinstance(l, list) and len(l) > 1
              and str(l[1]).endswith(".Cu")]
    return {"footprints": fps, "track_widths": dict(widths), "has_stackup": stackup is not None,
            "copper_layers": layers, "version": (first(root, "version") or [None, None])[1],
            "generator": (first(root, "generator") or [None, None])[1]}


def floating(n):
    """KiCad names nets of pins with nothing attached 'unconnected-(...)' - electrically floating."""
    return n is None or str(n).startswith("unconnected-")


def is_gnd(n):
    return bool(n) and re.search(r"\bGND\b|^/?GND|VSS|^0V$", n, re.I) is not None


def review(board: dict) -> dict:
    out = {"errors": [], "warnings": [], "info": []}
    E, W, I = out["errors"].append, out["warnings"].append, out["info"].append
    fps = board["footprints"]
    I(f"KiCad file version {board['version']} generator {board['generator']}; copper layers: {board['copper_layers']}")
    if not board["has_stackup"]:
        W("No board stackup defined in Board Setup: impedance/current capacity cannot be assessed.")
    imu = [f for f in fps if re.search(r"LSM6DS3", f["value"] + f["lib"], re.I)]
    hdr = [f for f in fps if re.search(r"raspberry|2x20|02x20|GPIO.?40|RPi", f["value"] + f["lib"] + f["ref"], re.I)
           and len(f["pads"]) >= 40]
    if not hdr:
        E("No 40-pin Raspberry Pi header footprint found (looked for 2x20 / RaspberryPi in value/footprint).")
        h = {}
    else:
        h = hdr[0]["pads"]
        I(f"Pi header: {hdr[0]['ref']} ({hdr[0]['lib']})")
        v5 = {h.get(p) for p in PI_5V}
        v33 = {h.get(p) for p in PI_3V3}
        gnds = {h.get(p) for p in PI_GND}
        if None in v5 or len(v5) != 1:
            W(f"Pi 5V pins 2/4 not both on one net: {[h.get(p) for p in PI_5V]} (needed if the board powers the Pi).")
        if None in v33 or len(v33) != 1:
            W(f"Pi 3V3 pins 1/17 nets: {[h.get(p) for p in PI_3V3]}")
        if None in gnds:
            W(f"Some Pi GND pins unconnected: {[p for p in PI_GND if h.get(p) is None]} - connect all GND pins for return current.")
        if len(gnds - {None}) > 1:
            E(f"Pi GND pins on different nets: {sorted(gnds - {None})}")
        if v5 & v33 - {None}:
            E("Pi 5V and 3V3 pins are on the SAME net: short circuit.")
        if (v5 | v33) & (gnds - {None}):
            E("A Pi power pin shares a net with GND: short circuit.")
        for p in PI_ID:
            if h.get(p) and not re.search(r"ID_S", h[p] or "", re.I):
                W(f"Pi pin {p} (ID_SD/ID_SC, reserved for HAT EEPROM) is used for net {h[p]}.")
        # who drives 5V?
        n5 = next(iter(v5 - {None}), None)
        if n5:
            drivers = [f"{f['ref']}({f['value']})" for f in fps if f is not hdr[0] and n5 in f["pads"].values()]
            I(f"Parts on Pi 5V net '{n5}': {drivers or 'none'}")
            if not drivers:
                I("Nothing on the board connects to the Pi 5V net: Pi must be powered through its own micro-USB.")
            else:
                I("If the board feeds the Pi through header 5V, that bypasses the Pi 3 input polyfuse/TVS: "
                  "the regulator must deliver >= 2.5 A clean 5.0-5.25 V; add reverse-polarity protection.")
            w = board["track_widths"].get(n5)
            if drivers and w and min(w) < 0.5:
                W(f"Narrowest track on 5V net '{n5}' is {min(w)} mm; for ~2.5 A on 1 oz copper use >= ~1.0 mm "
                  "(check with an IPC-2152 calculator) or a pour.")
    if not imu:
        E("No LSM6DS3TR-C footprint found.")
        return out
    for u in imu:
        p = u["pads"]
        ref = u["ref"]
        I(f"IMU {ref}: " + ", ".join(f"{k}:{IMU_PINS.get(k, '?')}={v}" for k, v in sorted(p.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99)))
        vdd, vddio = p.get("8"), p.get("5")
        gnd_nets = {p.get("6"), p.get("7")}
        if vdd is None or vddio is None:
            E(f"{ref}: VDD (pin 8) or VDDIO (pin 5) unconnected.")
        for n, nm in ((vdd, "VDD"), (vddio, "VDDIO")):
            if n and re.search(r"5V|VBUS|VIN|VBAT|12V", n, re.I) and not re.search(r"3V3|3\.3", n):
                E(f"{ref}: {nm} on net '{n}' - LSM6DS3TR-C max supply is 3.6 V (abs max 4.8 V). Use 3V3.")
            if h and n and n in {h.get(x) for x in PI_5V}:
                E(f"{ref}: {nm} tied to Pi 5V - will damage the IMU.")
        if h and vddio and vddio not in {h.get(x) for x in PI_3V3}:
            W(f"{ref}: VDDIO net '{vddio}' is not the Pi 3V3 net; Pi GPIO are 3.3 V, so VDDIO must be 3.3 V for I2C levels.")
        if None in gnd_nets or not all(is_gnd(n) for n in gnd_nets):
            E(f"{ref}: GND pins 6/7 nets {sorted(map(str, gnd_nets))} are not both ground.")
        if h and gnd_nets - {None} and not (gnd_nets & {h.get(x) for x in PI_GND}):
            E(f"{ref}: IMU ground is not the Pi ground net - no common reference.")
        cs, sa0 = p.get("12"), p.get("1")
        spi_hdr = {h.get(k): k for k in PI_SPI0 if h and not floating(h.get(k))}
        spi_mode = bool(h) and cs in spi_hdr
        if spi_mode:
            I(f"{ref}: wired for 4-wire SPI (CS on Pi header pin {spi_hdr[cs]} {PI_SPI0[spi_hdr[cs]]}). "
              "Datasheet-valid. Software must use SPI (spidev), not I2C.")
            want = {"13": "23", "14": "19", "1": "21"}   # SPC->SCLK, SDI->MOSI, SDO->MISO
            for ipin, hpin in want.items():
                if p.get(ipin) != h.get(hpin):
                    E(f"{ref}: pin {ipin} ({IMU_PINS[ipin]}) net '{p.get(ipin)}' != Pi header pin {hpin} "
                      f"({PI_SPI0[hpin]}) net '{h.get(hpin)}'.")
                else:
                    I(f"{ref}: pin {ipin} {IMU_PINS[ipin]} -> header pin {hpin} {PI_SPI0[hpin]} OK")
        else:
            if floating(cs):
                W(f"{ref}: CS (pin 12) floating. Internal pull-up selects I2C, but tie CS to VDDIO for a defined mode.")
            elif is_gnd(cs):
                E(f"{ref}: CS tied to GND -> SPI mode, I2C disabled, and no Pi SPI0 chip-select wired: interface is dead.")
            elif cs != vddio:
                W(f"{ref}: CS on net '{cs}' (expected VDDIO for I2C or a Pi SPI0 CE pin).")
            if floating(sa0):
                E(f"{ref}: SDO/SA0 (pin 1) floating - in I2C mode it is an input without pull-up; address undefined.")
            elif is_gnd(sa0):
                I(f"{ref}: SA0=GND -> I2C address 0x6A")
            elif sa0 == vddio:
                I(f"{ref}: SA0=VDDIO -> I2C address 0x6B (set imu_address=107 in config)")
            if h:
                if p.get("13") != h.get(PI_SCL1):
                    E(f"{ref}: SCL (pin 13) net '{p.get('13')}' != Pi GPIO3/SCL1 (header pin 5) net '{h.get(PI_SCL1)}'.")
                if p.get("14") != h.get(PI_SDA1):
                    E(f"{ref}: SDA (pin 14) net '{p.get('14')}' != Pi GPIO2/SDA1 (header pin 3) net '{h.get(PI_SDA1)}'.")
                else:
                    I("Pi 3 already has 1.8 kOhm pull-ups to 3V3 on GPIO2/3; extra board pull-ups are optional.")
        for pin in ("2", "3"):
            n = p.get(pin)
            if floating(n):
                E(f"{ref}: {IMU_PINS[pin]} (pin {pin}) is floating (net '{n}'). ST DS Table 2 / AN5130 sec 3.6: in Mode 1 "
                  "SDx/SCx must be connected to VDDIO or GND and cannot be left floating (recommended: VDDIO). "
                  "A schematic no-connect flag hides this from ERC.")
            elif not (is_gnd(n) or n == vddio):
                W(f"{ref}: {IMU_PINS[pin]} on '{n}' - only valid if you use sensor-hub Mode 2.")
        for pin in ("10", "11"):
            if not floating(p.get(pin)):
                W(f"{ref}: pin {pin} is NC (internal pull-up) but connected to '{p[pin]}'; datasheet says leave unconnected.")
        caps = [f for f in fps if f["ref"].upper().startswith("C")]
        for n, nm in ((vdd, "VDD"), (vddio, "VDDIO")):
            if not n:
                continue
            dec = [c for c in caps if n in c["pads"].values() and any(is_gnd(x) for x in c["pads"].values())]
            if not dec:
                E(f"{ref}: no decoupling capacitor between {nm} net '{n}' and GND (datasheet: 100 nF).")
            elif not any(re.search(r"100\s*n|0?\.1\s*u|104", c["value"], re.I) for c in dec):
                W(f"{ref}: {nm} decoupling caps {[c['ref'] + '=' + c['value'] for c in dec]} - none is 100 nF.")
            else:
                pin = "8" if nm == "VDD" else "5"
                src = u["pos"].get(pin)
                dists = sorted((min(math.dist(src, q) for q in c["pos"].values()), c["ref"], c["value"])
                               for c in dec if src and c["pos"])
                txt = ", ".join(f"{r}={v} at {d:.1f} mm" for d, r, v in dists)
                if dists and dists[0][0] > 3.0:
                    W(f"{ref}: nearest decoupling cap to {nm} (pin {pin}) is {dists[0][0]:.1f} mm away ({txt}). "
                      "Datasheet recommends a 100 nF cap on each of VDD and VDDIO placed as close as possible to the pins "
                      "(aim < ~2-3 mm, short direct path to GND).")
                else:
                    I(f"{ref}: {nm} decoupled by {txt}")
    usb = [f for f in fps if re.search(r"USB", f["value"] + f["lib"], re.I)]
    for u in usb:
        I(f"USB footprint {u['ref']} ({u['lib']}) nets: {sorted(set(map(str, u['pads'].values())))}")
        names = " ".join(map(str, u["pads"].values()))  # noqa
        if re.search(r"D\+|DP", names) and not re.search(r"D-|DM|DN", names):
            W(f"{u['ref']}: D+ routed but no D- net.")
    if not usb:
        I("No USB footprint on the PCB: the USB IR camera plugs straight into a Pi 3 USB-A port (fine). "
          "Pi 3 USB ports share ~1.2 A total; check the camera's current draw.")
    unconnected = [(f["ref"], k) for f in fps for k, v in f["pads"].items() if floating(v)
                   and not (f in imu and k in ("10", "11", "4", "9"))]
    if unconnected:
        I(f"{len(unconnected)} pads have no net or an unconnected-(...) net (unused header pins are expected); first 20: {unconnected[:20]}")
    return out


def main():
    p = Path(sys.argv[1])
    r = review(load_board(p.read_text(errors="replace")))
    if "--json" in sys.argv:
        print(json.dumps(r, indent=2))
    else:
        for k in ("errors", "warnings", "info"):
            print(f"== {k.upper()} ({len(r[k])})")
            for x in r[k]:
                print("  -", x)
    sys.exit(1 if r["errors"] else 0)


if __name__ == "__main__":
    main()
