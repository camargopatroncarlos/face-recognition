import cv2
import face_recognition
import os

KNOWN_FACES_DIR = "data/known"
TOLERANCE = 0.5

known_encodings = []
known_names = []

# Load known faces
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)

    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]

        known_encodings.append(encoding)
        known_names.append(name)

print("Known faces loaded:", known_names)

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change index if needed



# --- performance knobs ---
FRAME_SCALE = 0.25      # 0.25 = 1/4 size (fast). Try 0.5 if too blurry
PROCESS_EVERY_N = 3     # do recognition every 3 frames

frame_count = 0
last_results = []  # cache results for frames we skip

while True:
    ret, frame_bgr = video_capture.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY_N == 0:
        small_frame_bgr = cv2.resize(frame_bgr, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        small_frame_rgb = cv2.cvtColor(small_frame_bgr, cv2.COLOR_BGR2RGB)

        face_locations_small = face_recognition.face_locations(small_frame_rgb, model="hog")
        face_encodings = face_recognition.face_encodings(small_frame_rgb, face_locations_small)

        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations_small, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]

            # scale boxes back up to original size
            scale = int(1 / FRAME_SCALE)
            results.append((top * scale, right * scale, bottom * scale, left * scale, name))

        last_results = results

    # Draw cached results on every frame
    for top, right, bottom, left, name in last_results:
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame_bgr, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

#         matches = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
#         name = "Unknown"

#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_names[first_match_index]

#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left, top - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

video_capture.release()
cv2.destroyAllWindows()