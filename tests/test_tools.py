from fixtures.make_pcb_fixtures import bad, good
from inspect_model_source import inspect
from pcb_review import load_board, review

BINARY_KERAS = '''
import tensorflow as tf
from tensorflow.keras import layers
train = tf.keras.utils.image_dataset_from_directory("fires/", image_size=(128,128))
m = tf.keras.Sequential([layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(), layers.Flatten(),
                         layers.Dense(64, activation="relu"), layers.Dense(1, activation="softmax")])
m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
m.fit(train, epochs=10)
'''

GOOD_KERAS = '''
import tensorflow as tf
from tensorflow.keras import layers
train = tf.keras.utils.image_dataset_from_directory("data/", validation_split=0.2, subset="training", seed=1)
m = tf.keras.Sequential([layers.Conv2D(16, 3), layers.Flatten(), layers.Dense(4, activation="softmax")])
m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# classes: no_fire, low, moderate, high
m.save("fire.keras")
'''


def test_inspect_flags_binary_softmax_bug():
    r = inspect(BINARY_KERAS)
    assert r["framework"] == "tensorflow/keras"
    assert r["final_layer"]["units"] == 1 and r["has_severity_output"] is False
    text = " ".join(r["findings"])
    assert "softmax over 1 unit" in text and "No model save" in text and "No negative" in text
    assert "No validation" in text


def test_inspect_good_model():
    r = inspect(GOOD_KERAS)
    assert r["has_severity_output"] is True and r["saves_weights"]
    assert not any("BUG" in f for f in r["findings"])


def test_inspect_syntax_error():
    assert inspect("def (:")["syntax_ok"] is False


def test_pcb_review_good_board():
    r = review(load_board(good()))
    assert r["errors"] == [], r["errors"]
    assert any("0x6A" in i for i in r["info"])


def test_pcb_review_bad_board():
    r = review(load_board(bad()))
    e = " ".join(r["errors"])
    assert "max supply is 3.6 V" in e or "Pi 5V" in e
    assert "SDO/SA0" in e and "SDx" in e
    assert "SCL (pin 13)" in e and "SDA (pin 14)" in e
    assert "CS tied to GND" in e
    assert "decoupling" in e


def test_pcb_review_spi_wiring_accepted():
    from fixtures.make_pcb_fixtures import spi_board
    r = review(load_board(spi_board()))
    assert r["errors"] == [], r["errors"]
    assert any("4-wire SPI" in i for i in r["info"])


def test_pcb_review_flags_floating_sdx_scx():
    from fixtures.make_pcb_fixtures import spi_floating_sdx
    e = " ".join(review(load_board(spi_floating_sdx()))["errors"])
    assert "SDx (pin 2) is floating" in e and "SCx (pin 3) is floating" in e
