"""SYNTHETIC KiCad 8-format .kicad_pcb fixtures for testing tools/pcb_review.py (not the user's board)."""

HDR = {1: "+3V3", 17: "+3V3", 2: "+5V", 4: "+5V", 3: "/SDA", 5: "/SCL",
       **{p: "GND" for p in (6, 9, 14, 20, 25, 30, 34, 39)}}


def fp(lib, ref, val, pads):
    ps = "\n".join(f'    (pad "{n}" smd rect (at 0 0) (size 1 1) (layers "F.Cu")' +
                   (f' (net 1 "{net}"))' if net else ")") for n, net in pads.items())
    return f'  (footprint "{lib}"\n    (property "Reference" "{ref}")\n    (property "Value" "{val}")\n{ps}\n  )'


def board(imu_pads, caps, extra=""):
    hdr = {str(i): HDR.get(i) for i in range(1, 41)}
    parts = [fp("Connector_PinSocket_2.54mm:PinSocket_2x20_P2.54mm_Vertical", "J1", "RaspberryPi_GPIO", hdr),
             fp("Package_LGA:LGA-14_3x2.5mm_P0.5mm_LayoutBorder3x4y", "U1", "LSM6DS3TR-C", imu_pads)]
    parts += [fp("Capacitor_SMD:C_0402_1005Metric", r, v, {"1": a, "2": b}) for r, v, a, b in caps]
    return ('(kicad_pcb (version 20240108) (generator "pcbnew")\n  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))\n'
            '  (setup (stackup (layer "F.Cu" (type "copper"))))\n' + "\n".join(parts) + extra + "\n)\n")


GOOD_IMU = {"1": "GND", "2": "+3V3", "3": "+3V3", "4": None, "5": "+3V3", "6": "GND", "7": "GND", "8": "+3V3",
            "9": None, "10": None, "11": None, "12": "+3V3", "13": "/SCL", "14": "/SDA"}
GOOD_CAPS = [("C1", "100nF", "+3V3", "GND"), ("C2", "100nF", "+3V3", "GND")]

BAD_IMU = {**GOOD_IMU, "8": "+5V", "1": None, "2": None, "13": "/SDA", "14": "/SCL", "12": "GND"}


def good():
    return board(GOOD_IMU, GOOD_CAPS)


def bad():
    return board(BAD_IMU, [])


# SPI-wired variant (same topology as an LSM6DS3 on SPI0/CE0), SYNTHETIC.
SPI_HDR = {19: "/MOSI", 21: "/MISO", 23: "/SCLK", 24: "/CE0", 11: "/GPIO17"}
SPI_IMU = {"1": "/MISO", "2": "+3V3", "3": "+3V3", "4": "/GPIO17", "5": "+3V3", "6": "GND", "7": "GND", "8": "+3V3",
           "9": None, "10": None, "11": None, "12": "/CE0", "13": "/SCLK", "14": "/MOSI"}


def spi_board(imu=None):
    global HDR
    saved = dict(HDR)
    HDR = {**{k: v for k, v in HDR.items() if k not in (3, 5)}, **SPI_HDR}
    try:
        return board(imu or SPI_IMU, GOOD_CAPS)
    finally:
        HDR = saved


def spi_floating_sdx():
    return spi_board({**SPI_IMU, "2": "unconnected-(U2-SDX-Pad2)", "3": "unconnected-(U2-SCX-Pad3)"})
