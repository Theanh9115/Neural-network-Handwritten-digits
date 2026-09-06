import numpy as np

from src.predict import drawing_to_input


def test_drawing_to_input_downsamples_and_normalizes():
    drawing = np.zeros((280, 280), dtype=np.uint8)
    drawing[100:110, 100:110] = 255

    result = drawing_to_input(drawing)

    assert result.shape == (784, 1)
    assert result.max() == 1.0
    assert result.min() == -1.0
    assert result[14 * 28 + 14, 0] == 1.0


def test_drawing_to_input_centers_small_digit():
    drawing = np.zeros((280, 280), dtype=np.uint8)
    drawing[20:40, 30:50] = 255

    result = drawing_to_input(drawing).reshape(28, 28)

    assert np.argmax(result) // 28 in range(4, 24)
    assert np.argmax(result) % 28 in range(4, 24)
