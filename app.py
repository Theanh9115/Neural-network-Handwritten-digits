import tkinter as tk

import numpy as np

from src.predict import predict_digit, load_model


CANVAS_SIZE = 280
BRUSH_RADIUS = 10


class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        self.drawing = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self.last_point = None
        self.model = load_model()

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background="black",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.canvas.pack(padx=16, pady=(16, 8))
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        controls = tk.Frame(root)
        controls.pack(pady=(0, 8))
        tk.Button(controls, text="Predict", command=self.predict).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(controls, text="Clear", command=self.clear).pack(
            side=tk.LEFT, padx=4
        )

        self.result = tk.Label(
            root,
            text="Draw a digit, then click Predict",
            font=("TkDefaultFont", 12),
        )
        self.result.pack(pady=(0, 16))

    def start_stroke(self, event):
        self.last_point = (event.x, event.y)
        self.paint(event.x, event.y)

    def draw_stroke(self, event):
        if self.last_point is None:
            self.start_stroke(event)
            return

        previous_x, previous_y = self.last_point
        self.canvas.create_line(
            previous_x,
            previous_y,
            event.x,
            event.y,
            fill="white",
            width=BRUSH_RADIUS * 2,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.paint_line(previous_x, previous_y, event.x, event.y)
        self.last_point = (event.x, event.y)

    def end_stroke(self, _event):
        self.last_point = None

    def paint(self, x, y):
        self.paint_line(x, y, x, y)

    def paint_line(self, start_x, start_y, end_x, end_y):
        distance = max(abs(end_x - start_x), abs(end_y - start_y), 1)
        for step in range(distance + 1):
            fraction = step / distance
            x = round(start_x + (end_x - start_x) * fraction)
            y = round(start_y + (end_y - start_y) * fraction)
            y_min = max(0, y - BRUSH_RADIUS)
            y_max = min(CANVAS_SIZE, y + BRUSH_RADIUS + 1)
            x_min = max(0, x - BRUSH_RADIUS)
            x_max = min(CANVAS_SIZE, x + BRUSH_RADIUS + 1)
            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= BRUSH_RADIUS**2
            self.drawing[y_min:y_max, x_min:x_max][mask] = 255

    def clear(self):
        self.canvas.delete("all")
        self.drawing.fill(0)
        self.result.config(text="Draw a digit, then click Predict")

    def predict(self):
        if not np.any(self.drawing):
            self.result.config(text="Draw a digit before predicting")
            return

        digit, score = predict_digit(self.model, self.drawing)
        self.result.config(text=f"Prediction: {digit}  (score: {score:.1%})")


if __name__ == "__main__":
    root = tk.Tk()
    DigitRecognizer(root)
    root.mainloop()
