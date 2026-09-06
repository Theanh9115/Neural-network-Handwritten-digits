# Handwritten Digit Recognizer

The project includes a small desktop app where you can draw a digit and ask
the trained neural network to predict it.

From this directory, run:

```text
python app.py
```

Draw one digit with the mouse, then select **Predict**. Select **Clear** to
try another digit.

## Training

The model is trained on MNIST. To train it for more epochs, open
`src/train.py`, increase `num_epochs` (for example, from `10` to `20`), and
run this command from the project directory:

```text
python src/train.py
```

Training resumes from `trained_model.npz` and replaces it when complete.
The drawing input is automatically cropped, resized, and centered to match
the MNIST image format before prediction.
