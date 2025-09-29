import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageOps
from digit_classification import Network
#allows us to directly draw digits, and test the neural network
#load previously trained model 
net = Network.load("trained_network.pkl")

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Identifier")

        top = tk.Frame(root)
        top.pack(padx=10, pady=10)

        # drawing canvas (large)
        self.canvas = tk.Canvas(top, width=280, height=280, bg="white")
        self.canvas.pack(side=tk.LEFT)

        # right column: preview + controls
        right = tk.Frame(top)
        right.pack(side=tk.LEFT, padx=10)

        # preview of the 28x28 image (upscaled)
        self.preview_label = tk.Label(right, text="Preview (28×28)")
        self.preview_label.pack()
        self.preview_img_label = tk.Label(right)
        self.preview_img_label.pack(pady=6)

        # buttons
        btns = tk.Frame(right)
        btns.pack(pady=6)
        tk.Button(btns, text="Predict", command=self.predict_digit).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=4)

        # result
        self.result = tk.Label(right, text="Draw a digit", font=("Arial", 16))
        self.result.pack(pady=10)

        #draw support
        self.canvas.bind("<B1-Motion>", self.draw)

        #internal image buffer
        self.image = Image.new("L", (280, 280), 255)

    def draw(self, event):
        r = 3  #brush size
        x1, y1, x2, y2 = event.x-r, event.y-r, event.x+r, event.y+r
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        for i in range(max(0, x1), min(280, x2)):
            for j in range(max(0, y1), min(280, y2)):
                self.image.putpixel((i, j), 0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.result.config(text="Draw a digit")
        self.preview_img_label.config(image="")

    def predict_digit(self):
        #resize to mnist
        # invert because user draws black-on-white but MNIST expects bright digit on dark bg
        processed = ImageOps.invert(self.image)
                # resize to 28x28 (use high-quality resampling)
        small = processed.resize((28, 28), Image.LANCZOS)

        # show a clear preview (upscale using NEAREST so the 28x28 pixels are visible)
        preview = small.resize((140, 140), Image.NEAREST)
        tk_preview = ImageTk.PhotoImage(preview)
        self.preview_img_label.config(image=tk_preview)
        self.preview_img_label.image = tk_preview  # keep reference

        # convert to vector expected by your network: shape (784,1), values in [0,1]
        arr = np.array(small).astype(np.float32) / 255.0
        vec = arr.reshape(784, 1)

        # --- predict ---
        output = net.get_output(vec)
        guess = int(np.argmax(output))

        self.result.config(text=f"Prediction: {guess}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
