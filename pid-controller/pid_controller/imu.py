"""Minimal LSM6DS3TR-C driver over I2C (smbus2). Register map per ST datasheet DocID030071."""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

WHO_AM_I = 0x0F
WHO_AM_I_VALUE = 0x6A
CTRL1_XL = 0x10
CTRL2_G = 0x11
CTRL3_C = 0x12
OUTX_L_G = 0x22   # 6 bytes gyro X/Y/Z, then 6 bytes accel at 0x28
OUTX_L_XL = 0x28

# 104 Hz ODR, accel +-4 g, gyro +-500 dps.
CTRL1_XL_104HZ_4G = 0x48   # ODR_XL=0100, FS_XL=10 (+-4g)
CTRL2_G_104HZ_500DPS = 0x44  # ODR_G=0100, FS_G=01 (500 dps)
CTRL3_C_BDU_IFINC = 0x44     # BDU=1, IF_INC=1
ACCEL_SENS_G_PER_LSB = 0.122e-3   # +-4 g
GYRO_SENS_DPS_PER_LSB = 17.50e-3  # 500 dps


@dataclass
class ImuSample:
    gyro_x: float
    gyro_y: float
    gyro_z: float
    accel_x: float
    accel_y: float
    accel_z: float

    @property
    def gyro_magnitude(self) -> float:
        return math.sqrt(self.gyro_x ** 2 + self.gyro_y ** 2 + self.gyro_z ** 2)


CTRL4_C = 0x13
CTRL4_C_I2C_DISABLE = 0x04   # recommended when the part is used on SPI (stops SPI traffic being seen as I2C)


def _configure(write):
    write(CTRL3_C, CTRL3_C_BDU_IFINC)
    write(CTRL1_XL, CTRL1_XL_104HZ_4G)
    write(CTRL2_G, CTRL2_G_104HZ_500DPS)


def _decode(raw: bytes) -> "ImuSample":
    gx, gy, gz, ax, ay, az = struct.unpack("<6h", bytes(raw))
    return ImuSample(gx * GYRO_SENS_DPS_PER_LSB, gy * GYRO_SENS_DPS_PER_LSB, gz * GYRO_SENS_DPS_PER_LSB,
                     ax * ACCEL_SENS_G_PER_LSB, ay * ACCEL_SENS_G_PER_LSB, az * ACCEL_SENS_G_PER_LSB)


class LSM6DS3TRC_SPI:
    """4-wire SPI, as wired on the Hackathon_Fall26 HAT: CS->GPIO8/CE0 (pin 24), SPC->GPIO11/SCLK (23),
    SDI->GPIO10/MOSI (19), SDO->GPIO9/MISO (21). Enable SPI first: sudo raspi-config nonint do_spi 0
    (device /dev/spidev0.0). SPI mode 3, <= 10 MHz per datasheet; 1 MHz default here."""

    def __init__(self, bus_num: int = 0, device: int = 0, max_speed_hz: int = 1_000_000, spi=None):
        if spi is None:
            import spidev  # pip install spidev (or apt install python3-spidev)
            spi = spidev.SpiDev()
            spi.open(bus_num, device)
            spi.max_speed_hz = max_speed_hz
            spi.mode = 0b11
        self.spi = spi
        who = self._read(WHO_AM_I, 1)[0]
        if who != WHO_AM_I_VALUE:
            raise RuntimeError(f"LSM6DS3TR-C not found on SPI{bus_num}.{device}: WHO_AM_I=0x{who:02X}")
        self._write(CTRL4_C, CTRL4_C_I2C_DISABLE)
        _configure(self._write)

    def _read(self, reg: int, n: int) -> list[int]:
        return self.spi.xfer2([0x80 | reg] + [0] * n)[1:]   # bit7=1 -> read; IF_INC auto-increments

    def _write(self, reg: int, val: int) -> None:
        self.spi.xfer2([reg & 0x7F, val])

    def read(self) -> "ImuSample":
        return _decode(self._read(OUTX_L_G, 12))


class LSM6DS3TRC:
    """I2C variant (for boards that tie CS to VDDIO and wire SDA/SCL to GPIO2/3). NOT how Hackathon_Fall26 is wired."""

    def __init__(self, bus_num: int = 1, address: int = 0x6A, bus=None):
        if bus is None:
            from smbus2 import SMBus  # imported lazily so the server side never needs it
            bus = SMBus(bus_num)
        self.bus = bus
        self.address = address
        who = self.bus.read_byte_data(address, WHO_AM_I)
        if who != WHO_AM_I_VALUE:
            raise RuntimeError(f"LSM6DS3TR-C not found at 0x{address:02X}: WHO_AM_I=0x{who:02X}")
        _configure(lambda r, v: self.bus.write_byte_data(address, r, v))

    def read(self) -> ImuSample:
        return _decode(self.bus.read_i2c_block_data(self.address, OUTX_L_G, 12))


def open_imu(interface: str, i2c_bus: int = 1, i2c_address: int = 0x6A, spi_bus: int = 0, spi_device: int = 0):
    if interface == "spi":
        return LSM6DS3TRC_SPI(spi_bus, spi_device)
    if interface == "i2c":
        return LSM6DS3TRC(i2c_bus, i2c_address)
    raise ValueError(f"imu_interface must be 'spi' or 'i2c', got {interface!r}")
