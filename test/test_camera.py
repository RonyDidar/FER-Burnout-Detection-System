### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      test_camera.py (Additional module for testing the camera's availability and setup)
### ========================================================================================================================================

# =============================================================================================
# SETUP
# =============================================================================================
import cv2

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================
# Setup the available camera index (In this case, it is 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Setup the frames and video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 20.0, (640, 480))

# Declare the iteration to ensure availability
print("Recording... press 'q' to stop.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not captured")
        break
    cv2.imshow("Preview", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop all the camera execution
cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved test.avi")

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================