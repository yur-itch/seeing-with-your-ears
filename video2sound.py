import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import image2sound
import tempfile
import os
import hilbert

def stitch_audio_with_crossfade(buffers, sample_rate=44100, fade_ms=2):
    fade_samples = int(sample_rate * fade_ms / 1000)
    if not buffers:
        return np.array([])
    
    # 1. Start with the first buffer
    # We will build a list of arrays to concatenate later (faster than repeated concatenation)
    output_parts = [buffers[0]]
    
    for buf in buffers[1:]:
        if fade_samples > 0:
            # Get the tail of the previous part
            prev_tail = output_parts[-1][-fade_samples:]
            # Get the head of the current part
            curr_head = buf[:fade_samples]
            
            # Create fade curves
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)
            
            # Mix the overlap
            mixed_part = prev_tail * fade_out + curr_head * fade_in
            
            # Trim the raw tail from the previous buffer in the list
            output_parts[-1] = output_parts[-1][:-fade_samples]
            
            # Add the mixed overlap
            output_parts.append(mixed_part)
            
            # Add the rest of the current buffer
            output_parts.append(buf[fade_samples:])
        else:
            output_parts.append(buf)

    return np.concatenate(output_parts)

def generate_video_with_sound(video_path, hilbert_iterations=4, frame_rate=None, volume=0.8,
                              output_path="bouncing_ball_with_sound.mp4", output_size=256):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if frame_rate is None or frame_rate <= 0:
        frame_rate = src_fps if src_fps > 0 else 60
    
    duration_per_frame = 1.0 / frame_rate
    
    # Generate slightly more audio per frame to account for the crossfade overlap
    fade_ms = 2
    fade_duration = fade_ms / 1000.0
    audio_generation_duration = duration_per_frame + fade_duration 

    print(f"Target FPS: {frame_rate}")
    print(f"Audio per frame: {audio_generation_duration:.4f}s (includes {fade_ms}ms overlap)")

    audio_buffers = []
    
    # Use .avi (MJPG) for temp file to avoid MP4 metadata corruption in OpenCV
    tmp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
    tmp_writer = None

    # Pre-calculate curve once
    hilbert_curve = hilbert.generate(hilbert_iterations)

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Process image for Sound (grayscale / resized)
        img_for_sound = image2sound.load_and_prepare_image_from_array(frame, hilbert_iterations)

        # 2. Process image for Video (Visuals)
        # Resize the PROCESSED image (img_for_sound) to output size
        # We must convert it back to BGR for the video writer
        out_frame = cv2.resize(img_for_sound, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_GRAY2BGR)

        # 3. Generate Sound
        samples = image2sound.generate_sound(
            img_for_sound, 
            hilbert_curve,
            sample_rate=44100,
            volume=volume,
            duration=audio_generation_duration 
        )
        audio_buffers.append(np.asarray(samples, dtype=np.float32))

        # 4. Initialize Writer if needed
        if tmp_writer is None:
            # MJPG is more robust for temp files than mp4v
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            tmp_writer = cv2.VideoWriter(tmp_video_file, fourcc, frame_rate, (output_size, output_size))

        tmp_writer.write(out_frame)
        frame_count += 1

    cap.release()
    if tmp_writer is not None:
        tmp_writer.release()

    print(f"Processed {frame_count} frames.")

    # --- Audio Processing ---
    if audio_buffers:
        full_audio = stitch_audio_with_crossfade(audio_buffers, sample_rate=44100, fade_ms=fade_ms)
        
        # Normalize
        max_val = np.max(np.abs(full_audio)) if full_audio.size else 0.0
        if max_val > 0:
            full_audio = (full_audio / max_val) * volume
        
        # Trim exact length to avoid float accumulation drift
        expected_samples = int(frame_count * duration_per_frame * 44100)
        full_audio = full_audio[:expected_samples]
        full_audio = full_audio.astype(np.float32)

        # Convert to Stereo (Mono AAC often causes "double duration" glitch in Windows)
        audio_array = full_audio.reshape(-1, 1)
        audio_array = np.tile(audio_array, (1, 2)) 

        audio_clip = AudioArrayClip(audio_array, fps=44100)
    else:
        audio_clip = None

    # --- Final Assembly ---
    # Load temp AVI
    video_clip = VideoFileClip(tmp_video_file)
    
    if audio_clip:
        # Strict duration matching
        audio_clip = audio_clip.with_duration(video_clip.duration)
        final_clip = video_clip.with_audio(audio_clip)
    else:
        final_clip = video_clip

    # Write final MP4
    # 'logger=None' silences the ffmpeg spam, remove if you want to see progress
    final_clip.write_videofile(
        output_path, 
        fps=frame_rate,
        codec='libx264', 
        audio_codec='aac',
        logger=None 
    )
    
    print(f"Done. Saved to {output_path}")

    # Cleanup
    video_clip.close()
    if audio_clip: audio_clip.close()
    final_clip.close()
    
    try:
        os.remove(tmp_video_file)
    except OSError:
        pass

if __name__ == "__main__":
    generate_video_with_sound(
        video_path="bouncing_ball.mp4",
        hilbert_iterations=5,
        frame_rate=60,
        volume=0.8,
        output_path="forest.mp4",
        output_size=256
    )