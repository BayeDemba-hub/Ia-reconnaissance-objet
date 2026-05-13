from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO("yolov8n.pt")

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    # Lire une frame
    ret, frame = cap.read()

    if not ret:
        break

    # Faire la détection
    results = model(frame)

    # Afficher les résultats annotés
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Quitter avec la touche q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
