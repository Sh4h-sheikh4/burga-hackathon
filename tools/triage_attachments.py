#!/usr/bin/env python3
"""Identify opaque attachments (.bin) by magic bytes / content and copy them with real extensions.
    python tools/triage_attachments.py <file-or-dir>... [--copy-to recovered/]
Recognises: zip (incl. Keras .keras, KiCad project archives), HDF5 (.h5 Keras weights), PyTorch (.pt zip/pickle),
ONNX, TFLite, KiCad .kicad_sch/.kicad_pcb/.kicad_pro, legacy EESchema, Gerber, Python source, images.
"""
import shutil
import sys
import zipfile
from pathlib import Path


def kind(p: Path) -> tuple[str, str]:
    b = p.read_bytes()[:4096]
    if b.startswith(b"\x89HDF\r\n\x1a\n"):
        return "hdf5 (Keras .h5 weights/model)", ".h5"
    if b[4:8] == b"TFL3":
        return "TFLite flatbuffer", ".tflite"
    if b.startswith(b"PK\x03\x04"):
        names = zipfile.ZipFile(p).namelist()
        if any(n.endswith("data.pkl") for n in names):
            return "PyTorch checkpoint (zip)", ".pt"
        if "config.json" in names and any(n.endswith(".weights.h5") for n in names):
            return "Keras v3 .keras", ".keras"
        if any(n.endswith((".kicad_pcb", ".kicad_sch", ".kicad_pro")) for n in names):
            return f"zip with KiCad project: {names[:20]}", ".zip"
        return f"zip: {names[:20]}", ".zip"
    if b.startswith(b"\x80") and b[1:2] in (b"\x02", b"\x03", b"\x04", b"\x05"):
        return "Python pickle (legacy torch.save / joblib)", ".pkl"
    t = b.decode("utf-8", "replace").lstrip()
    if t.startswith("(kicad_pcb"):
        return "KiCad PCB", ".kicad_pcb"
    if t.startswith("(kicad_sch"):
        return "KiCad schematic", ".kicad_sch"
    if t.startswith("(kicad_symbol_lib"):
        return "KiCad symbol library", ".kicad_sym"
    if t.startswith("(footprint") or t.startswith("(module"):
        return "KiCad footprint", ".kicad_mod"
    if t.startswith("EESchema Schematic"):
        return "KiCad 5 legacy schematic", ".sch"
    if t.startswith("{") and '"board"' in t[:2000]:
        return "KiCad project (.kicad_pro JSON)", ".kicad_pro"
    if t.startswith("(export") or t.startswith("(design"):
        return "KiCad netlist", ".net"
    if t.startswith("%FS") or t.startswith("G04"):
        return "Gerber", ".gbr"
    if b"\x08" in b[:16] and b"onnx" in b[:256].lower():
        return "ONNX model (probable)", ".onnx"
    if b.startswith(b"\xff\xd8"):
        return "JPEG", ".jpg"
    if b.startswith(b"\x89PNG"):
        return "PNG", ".png"
    if any(k in t for k in ("import ", "def ", "class ")):
        return "Python source", ".py"
    return "unknown", ".bin"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dest = Path(sys.argv[sys.argv.index("--copy-to") + 1]) if "--copy-to" in sys.argv else None
    if dest:
        args.remove(str(dest)) if str(dest) in args else None
        dest.mkdir(parents=True, exist_ok=True)
    files = [f for a in args for f in (Path(a).iterdir() if Path(a).is_dir() else [Path(a)]) if f.is_file()]
    for f in files:
        k, ext = kind(f)
        print(f"{f.name}: {k} ({f.stat().st_size} bytes)")
        if dest:
            shutil.copy2(f, dest / (f.stem[:16] + ext))  # originals are never modified


if __name__ == "__main__":
    main()
