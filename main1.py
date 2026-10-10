
import cv2

# ---------------------- CONFIG -----------------------
video_source = 'cars.mp4' # Change to 0 for webcam
cascade_path = 'cars.xml' # Path to Haar cascade file 
# --------------------------------------------------

# Capture video from file or webcam 
cap = cv2.VideoCapture(video_source)

# Load the pre-trained haar Cascade classifier 
car_cascade = cv2.CascadeClassifier(cascade_path)

if car_cascade.empty():
    print("Error: Could not load cascade classifier.")
    exit()

while True:
    ret, frames = cap.read()

    # Stop if no frame is captured (end of video or error)
    if not ret:
        print("End of video or cannot read from source.")
        break

    # Convert frame to grayscale 
    gray = cv2.cvtColor(frames, cv2.COLOR_BG2GRAY)

    #Detect cars 
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    # Draw rectangle & count cars   
    for (x, y, w, h) in cars: 
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)

    car_count = len(cars)
    cv2.putText(frames, f"Cars detected: {car_count}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # display the frame 
    cv2.imshow('Car Detection', frames)

    # Exit on Esc key 
    if cv2.waitkey(33) == 27:
        break 

# Release resources 
cap.release()
cv2.destroyAllWindows()

