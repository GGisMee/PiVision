from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from time import sleep
import subprocess

save_to = ['/media/gustavgamstedt/Samsung USB', '']
save_to = save_to[0]

# Initiera kameran
camera = Picamera2()

# Ställ in för att visa förhandsvisningen
camera.start_preview(Preview.QT)

# Konfigurera kameran för video
camera.configure(camera.create_video_configuration())

# Starta kameran
camera.start()

# Skapa encoder för H264-video
encoder = H264Encoder()

# Tillfällig H.264-utdatafil
h264_output_file = "video_test2.h264"
output = FileOutput(h264_output_file)

# Starta inspelningen med encoder och utdata
camera.start_recording(encoder, output)

# Filma i 10 sekunder
print("Recording started")
sleep(10)
print("Recording stopped")

# Stoppa inspelningen
camera.stop_recording()

# Konvertera H.264 till MP4 med ffmpeg
mp4_output_file = "video_test2.mp4"
subprocess.run([
    "ffmpeg", "-y", "-i", h264_output_file, "-c:v", "copy", mp4_output_file
])

print(f"Video saved as {mp4_output_file}")
