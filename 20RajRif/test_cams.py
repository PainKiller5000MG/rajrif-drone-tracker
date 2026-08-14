import cv2

CAM_INDEX = 1

print(f"Opening camera index {CAM_INDEX} ...")
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: could not open camera at index {CAM_INDEX}")
    exit(1)

print("Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Cam test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
