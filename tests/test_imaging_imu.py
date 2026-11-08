import numpy as np
from PIL import Image

from firedrone.config import InputSpec, Thresholds
from firedrone.imaging import decode, encode_jpeg, frame_stats, preprocess, quality_flags
from firedrone.imu import LSM6DS3TRC, OUTX_L_G
from firedrone.synthetic import synthetic_ir_frame


def test_preprocess_shapes():
    img = synthetic_ir_frame(10)
    assert preprocess(img, InputSpec()).shape == (1, 224, 224, 3)
    x = preprocess(img, InputSpec(layout="NCHW", scale="imagenet"))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert preprocess(img, InputSpec(channels=1)).shape == (1, 224, 224, 1)
    assert 0.0 <= preprocess(img, InputSpec()).min() and preprocess(img, InputSpec()).max() <= 1.0


def test_jpeg_roundtrip_and_proxy_grows_with_hotspot():
    t = Thresholds()
    small = frame_stats(decode(encode_jpeg(synthetic_ir_frame(5))), t)
    big = frame_stats(decode(encode_jpeg(synthetic_ir_frame(40))), t)
    none = frame_stats(decode(encode_jpeg(synthetic_ir_frame(0))), t)
    assert none.hot_pixel_fraction == 0 and small.hot_pixel_fraction < big.hot_pixel_fraction


def test_quality_flags():
    t = Thresholds()
    black = frame_stats(Image.new("L", (64, 64), 0), t)
    assert "too_dark" in quality_flags(black, t) and "blurry_or_flat" in quality_flags(black, t)
    assert "high_rotation_rate" in quality_flags(frame_stats(synthetic_ir_frame(10), t), t, gyro_dps=500)


class FakeBus:
    def __init__(self, raw):
        self.raw, self.writes = raw, {}

    def read_byte_data(self, a, r):
        return 0x6A

    def write_byte_data(self, a, r, v):
        self.writes[r] = v

    def read_i2c_block_data(self, a, r, n):
        assert r == OUTX_L_G and n == 12
        return self.raw


def test_imu_scaling():
    import struct
    raw = list(struct.pack("<6h", 1000, -1000, 0, 8197, 0, -8197))
    bus = FakeBus(raw)
    s = LSM6DS3TRC(bus=bus).read()
    assert bus.writes == {0x12: 0x44, 0x10: 0x48, 0x11: 0x44}
    assert abs(s.gyro_x - 17.5) < 1e-6 and abs(s.gyro_y + 17.5) < 1e-6
    assert abs(s.accel_x - 1.0) < 1e-3 and abs(s.accel_z + 1.0) < 1e-3


def test_imu_wrong_chip_raises():
    import pytest
    bus = FakeBus([0] * 12)
    bus.read_byte_data = lambda a, r: 0x69
    with pytest.raises(RuntimeError):
        LSM6DS3TRC(bus=bus)


class FakeSpi:
    """Emulates LSM6DS3TR-C SPI framing: first byte = address (bit7 read), then data."""

    def __init__(self, data12):
        self.regs = {0x0F: 0x6A}
        for i, b in enumerate(data12):
            self.regs[0x22 + i] = b
        self.writes = {}

    def xfer2(self, buf):
        addr = buf[0]
        if addr & 0x80:
            r = addr & 0x7F
            return [0] + [self.regs.get(r + i, 0) for i in range(len(buf) - 1)]
        self.writes[addr] = buf[1]
        return [0, 0]


def test_imu_spi_framing_and_scaling():
    import struct
    from firedrone.imu import LSM6DS3TRC_SPI
    spi = FakeSpi(list(struct.pack("<6h", 1000, 0, -1000, 0, 8197, 0)))
    s = LSM6DS3TRC_SPI(spi=spi).read()
    assert spi.writes == {0x13: 0x04, 0x12: 0x44, 0x10: 0x48, 0x11: 0x44}   # I2C disabled, BDU+IF_INC, ODR/FS
    assert abs(s.gyro_x - 17.5) < 1e-6 and abs(s.gyro_z + 17.5) < 1e-6 and abs(s.accel_y - 1.0) < 1e-3


def test_open_imu_rejects_bad_interface():
    import pytest
    from firedrone.imu import open_imu
    with pytest.raises(ValueError):
        open_imu("uart")
