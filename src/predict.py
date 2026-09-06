import os

import numpy as np

from src.neural_network import NeuralNetwork


IMAGE_SIZE = 28


def resize_image(image, height, width):
    """Resize an image with bilinear interpolation."""
    source_height, source_width = image.shape
    rows = np.linspace(0, source_height - 1, height)
    columns = np.linspace(0, source_width - 1, width)
    row_floor = np.floor(rows).astype(int)
    column_floor = np.floor(columns).astype(int)
    row_ceil = np.minimum(row_floor + 1, source_height - 1)
    column_ceil = np.minimum(column_floor + 1, source_width - 1)
    row_weight = rows - row_floor
    column_weight = columns - column_floor

    top_left = image[row_floor[:, None], column_floor]
    top_right = image[row_floor[:, None], column_ceil]
    bottom_left = image[row_ceil[:, None], column_floor]
    bottom_right = image[row_ceil[:, None], column_ceil]
    top = top_left * (1 - column_weight) + top_right * column_weight
    bottom = bottom_left * (1 - column_weight) + bottom_right * column_weight
    return top * (1 - row_weight[:, None]) + bottom * row_weight[:, None]


def load_model(model_path=None):
    """Load the saved handwritten-digit model."""
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "trained_model.npz",
        )

    data = np.load(model_path)
    model = NeuralNetwork(IMAGE_SIZE * IMAGE_SIZE, 16, 10)
    model._w1 = data["W1"]
    model._b1 = data["b1"]
    model._w2 = data["W2"]
    model._b2 = data["b2"]
    model._w3 = data["W3"]
    model._b3 = data["b3"]
    return model


def drawing_to_input(drawing):
    """Convert a square grayscale drawing into the model's normalized input."""
    pixels = np.asarray(drawing, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[0] != pixels.shape[1]:
        raise ValueError("drawing must be a square two-dimensional array")

    nonzero = np.argwhere(pixels > 0)
    if nonzero.size:
        top, left = nonzero.min(axis=0)
        bottom, right = nonzero.max(axis=0) + 1
        pixels = pixels[top:bottom, left:right]

        # MNIST digits occupy roughly a 20x20 box inside a 28x28 image.
        height, width = pixels.shape
        target_size = 20
        scale = target_size / max(height, width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        pixels = resize_image(pixels, resized_height, resized_width)

        centered = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
        top = (IMAGE_SIZE - resized_height) // 2
        left = (IMAGE_SIZE - resized_width) // 2
        centered[top:top + resized_height, left:left + resized_width] = pixels
        pixels = centered

        # MNIST centers the ink's center of mass rather than only its bounding box.
        total_ink = pixels.sum()
        if total_ink:
            rows, columns = np.indices(pixels.shape)
            center_row = int(round((rows * pixels).sum() / total_ink))
            center_column = int(round((columns * pixels).sum() / total_ink))
            shift_row = IMAGE_SIZE // 2 - center_row
            shift_column = IMAGE_SIZE // 2 - center_column
            pixels = np.roll(pixels, (shift_row, shift_column), axis=(0, 1))
    else:
        pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)

    pixels = np.clip(pixels / 255.0, 0.0, 1.0)
    return (pixels * 2.0 - 1.0).reshape(IMAGE_SIZE * IMAGE_SIZE, 1)


def predict_digit(model, drawing):
    """Return the most likely digit and its confidence-like output score."""
    output = model.forward(drawing_to_input(drawing))
    probabilities = output.ravel()
    digit = int(np.argmax(probabilities))
    return digit, float(probabilities[digit])
