from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

camera_config = picam2.video_configuration
# create_preview_configuration()
picam2.configure(camera_config)

#picam2.start_preview(Preview.QTGL)
picam2.start()


frame = picam2.capture_array()
print(frame)
picam2.stop()
picam2.close()
# picam2.capture_file("test_photo.jpg")