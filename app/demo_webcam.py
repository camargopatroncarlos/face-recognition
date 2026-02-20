import cv2

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("✅ Webcam opened. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow("Portfolio Demo - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()