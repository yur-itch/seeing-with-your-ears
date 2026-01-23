import numpy as np
import sounddevice as sd

samplerate = 44100
duration = 0.5  # seconds
freq = 440      # A4
t = np.linspace(0, duration, int(samplerate*duration), endpoint=False)
y = 0.3 * np.sin(2 * np.pi * freq * t)
sd.play(y, samplerate, blocking=False)
sd.wait()
