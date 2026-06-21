import cv2
import hilbert
import numpy as np
from scipy.io.wavfile import write

def load_and_prepare_image(path, hilbert_iterations):
    side_length = 2 ** hilbert_iterations

    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError("Could not open " + path)

    height, width = img.shape[:2]
    middle_y = height // 2
    middle_x = width // 2
    radius = min(middle_x, middle_y)
    
    img = img[
        middle_y-radius:middle_y+radius,
        middle_x-radius:middle_x+radius
    ]
    img = cv2.resize(img, (side_length, side_length))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)

    return img

# in image2sound.py
def load_and_prepare_image_from_array(frame_array, hilbert_iterations):
    """
    Resize a frame array (BGR) to 2^hilbert_iterations and convert to grayscale.
    """
    side_length = 2 ** hilbert_iterations
    height, width = frame_array.shape[:2]
    middle_y = height // 2
    middle_x = width // 2
    radius = min(middle_x, middle_y)
    
    img = frame_array[
        middle_y-radius:middle_y+radius,
        middle_x-radius:middle_x+radius
    ]
    img = cv2.resize(img, (side_length, side_length))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def perception_to_freq(t, min_freq, max_freq):
    return min_freq * (max_freq / min_freq) ** t

def get_frequencies(img, curve):
    min_freq = 200
    max_freq = 2000
    curve_arr = np.array(curve)                                                   # (N, 2)
    N = len(curve_arr)
    t = np.arange(N, dtype=np.float32) / max(1, N - 1)                           # (N,)
    freqs = (min_freq * (max_freq / min_freq) ** t).astype(np.float32)            # (N,)
    amps  = img[curve_arr[:, 0], curve_arr[:, 1]].astype(np.float32)             # (N,)
    return freqs, amps

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.float32)

def generate_silence(duration, sample_rate):
    num_samples = int(duration * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def generate_sound(img, curve, sample_rate=44100, volume=1, duration=10):
    freqs, amps = get_frequencies(img, curve)
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)    # (S,)
    # build all sine waves at once: (N, S), then collapse to (S,)
    waves = (amps / 255.0)[:, None] * np.sin(2 * np.pi * freqs[:, None] * t[None, :])
    samples = waves.sum(axis=0)
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples /= max_val
    samples *= volume
    return samples.astype(np.float32)

def precompute_sin_waves(curve, sample_rate=44100, duration=1.0/60):
    """Compute the (N, S) sine matrix once — identical for every frame of a video."""
    min_freq, max_freq = 200, 2000
    curve_arr = np.array(curve)                                                   # (N, 2)
    N = len(curve_arr)
    t_idx = np.arange(N, dtype=np.float32) / max(1, N - 1)                       # (N,)
    freqs = (min_freq * (max_freq / min_freq) ** t_idx).astype(np.float32)        # (N,)
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)    # (S,)
    sin_waves = np.sin(2 * np.pi * freqs[:, None] * t[None, :]).astype(np.float32)  # (N, S)
    return sin_waves, curve_arr

def generate_sound_batch(imgs, curve_arr, sin_waves, volume=1):
    """
    imgs:      list of B grayscale (H, W) uint8 arrays
    curve_arr: (N, 2) from precompute_sin_waves
    sin_waves: (N, S) from precompute_sin_waves
    Returns:   (B, S) float32, each frame normalised independently
    """
    imgs_stack = np.stack(imgs, axis=0)                                           # (B, H, W)
    amps = imgs_stack[:, curve_arr[:, 0], curve_arr[:, 1]].astype(np.float32)    # (B, N)
    samples = (amps / 255.0) @ sin_waves                                          # (B, S) matmul
    max_vals = np.max(np.abs(samples), axis=1, keepdims=True)                    # (B, 1)
    max_vals = np.where(max_vals > 0, max_vals, 1.0)
    return (samples / max_vals * volume).astype(np.float32)

def generate_hilbert_sound(img, hilbert_iterations, sample_rate=44100, volume=1, duration=10):
    curve = hilbert.generate(hilbert_iterations)
    samples = generate_sound(img, curve, sample_rate, volume, duration)
    return samples

def save_to_wav(filename, samples, sample_rate=44100):
    samples_int16 = np.int16(samples * 32767)
    write(filename, sample_rate, samples_int16)

def imshow_fixed_size(label, img, height, width):
    cv2.imshow(label, cv2.resize(img, (width, height)))


if __name__ == "__main__":
    hilbert_iterations = 4
    image_path = "samples/images/square.jfif"
    image = load_and_prepare_image(image_path, hilbert_iterations)
    samples = generate_hilbert_sound(image, hilbert_iterations)
    print(len(samples))
    imshow_fixed_size("Image", image, 512, 512)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    output_wav = "hilbert_sound.wav"
    save_to_wav(output_wav, samples)
    print(f"Saved audio to {output_wav}")
