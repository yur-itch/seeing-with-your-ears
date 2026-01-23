import cv2
import numpy as np

width, height = 256, 256
fps = 60
duration = 100
frames = fps * duration

radius = 20
position = np.array([width // 2, height // 2], dtype=float)
velocity = np.array([200, -300], dtype=float)
gravity = 500

# Create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('bouncing_ball.mp4', fourcc, fps, (width, height))

dt = 1 / fps

for _ in range(frames):
    velocity[1] += gravity * dt
    position += velocity * dt

    if position[0] - radius < 0:
        position[0] = radius
        velocity[0] *= -1
    if position[0] + radius > width:
        position[0] = width - radius
        velocity[0] *= -1
    if position[1] - radius < 0:
        position[1] = radius
        velocity[1] *= -1
    if position[1] + radius > height:
        position[1] = height - radius
        velocity[1] *= -1

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame, (int(position[0]), int(position[1])), radius, (255, 255, 255), -1)
    out.write(frame)

out.release()
print("Video saved as 'bouncing_ball.mp4'")
