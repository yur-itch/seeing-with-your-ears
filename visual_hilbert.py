import tkinter as tk
import colorsys
import numpy as np
import sounddevice as sd
from hilbert import generate

# ======================
# Audio setup
# ======================
SAMPLERATE = 44100
CHUNK_DURATION = 0.04   # 80 ms of audio
CALL_INTERVAL  = int(CHUNK_DURATION * 500)    # call every 40 ms (overlap!)
CHUNK_SAMPLES = int(SAMPLERATE * CHUNK_DURATION)

stream = sd.OutputStream(
    samplerate=SAMPLERATE,
    channels=1,
    dtype="float32",
)
stream.start()


# ======================
# Color helpers
# ======================
def interpolate_hsl(color1, color2, t):
    h1, l1, s1 = colorsys.rgb_to_hls(color1[0]/255, color1[1]/255, color1[2]/255)
    h2, l2, s2 = colorsys.rgb_to_hls(color2[0]/255, color2[1]/255, color2[2]/255)

    dh = h2 - h1
    if dh > 0.5:
        dh -= 1
    elif dh < -0.5:
        dh += 1

    h = (h1 + dh * t) % 1.0
    l = l1 + (l2 - l1) * t
    s = s1 + (s2 - s1) * t

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


# ======================
# Audio synthesis
# ======================
phase = 0.0
last_freq = None

def play_sine(t):
    global phase, last_freq

    # Map t → frequency
    F_MIN = 200
    F_MAX = 2000

    freq = F_MIN * (F_MAX / F_MIN) ** t
    freq = np.clip(freq, 200, 2000)

    # Generate sine wave chunk
    phase_inc = 2 * np.pi * freq / SAMPLERATE
    phases = phase + phase_inc * np.arange(CHUNK_SAMPLES)
    samples = 1 * np.sin(phases)

    phase = phases[-1] % (2 * np.pi)

    stream.write(samples.astype(np.float32).reshape(-1, 1))


# ======================
# Tkinter App
# ======================
class App(tk.Tk):
    def __init__(self, hilbert_order=7, width=500, height=500):
        super().__init__()
        self.title("Hilbert Sonifier")

        self.hilbert_order = hilbert_order
        self.width = width
        self.height = height

        self.side_len = 2 ** hilbert_order
        self.pixel_w = width / self.side_len
        self.pixel_h = height / self.side_len

        self.curve = generate(hilbert_order)
        self.t_map = {}

        self.mouse_down = False
        self.current_t = None

        self.canvas = tk.Canvas(self, width=width, height=height)
        self.canvas.pack()

        self.draw_curve()

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)

    def draw_curve(self):
        for i, (x, y) in enumerate(self.curve):
            t = i / max(len(self.curve) - 1, 1)
            color = interpolate_hsl((255, 0, 0), (0, 255, 255), t)
            self.canvas.create_rectangle(
                x * self.pixel_w,
                y * self.pixel_h,
                (x + 1) * self.pixel_w,
                (y + 1) * self.pixel_h,
                fill=color,
                outline=color
            )
            self.t_map[(x, y)] = t

    def on_mouse_down(self, event):
        self.mouse_down = True
        self.update_t(event)
        self.audio_loop()

    def on_mouse_up(self, event):
        self.mouse_down = False

    def on_mouse_move(self, event):
        if self.mouse_down:
            self.update_t(event)

    def update_t(self, event):
        gx = int(event.x / self.pixel_w)
        gy = int(event.y / self.pixel_h)
        key = (gx, gy)
        if key in self.t_map:
            self.current_t = self.t_map[key]

    def audio_loop(self):
        if self.mouse_down and self.current_t is not None:
            play_sine(self.current_t)
            self.after(CALL_INTERVAL, self.audio_loop)



# ======================
# Run
# ======================
if __name__ == "__main__":
    app = App(hilbert_order=4)
    app.mainloop()
    stream.stop()
    stream.close()
