import cv2
import numpy as np
from person_counter import PersonCounter

# YOLOv4 ağırlıklarını ve yapılandırma dosyalarını yükleyin
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Sınıf isimlerini yükleyin
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Poligonun köşe noktaları
points = np.array([(0, 358), (797, 0), (1275, 0), (1163, 715), (1, 718)])

# Video akışını başlatın
cap = cv2.VideoCapture("video.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video kaydediciyi başlatın
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 30.0, (frame_width, frame_height))

person_counter = PersonCounter()

while True:
    # Kameradan bir kare alnıyor
    ret, frame = cap.read()

    # Poligonun çevresini işaretle
    cv2.polylines(frame, [points], True, (0, 0, 255), 2)

    # YOLOv4 için girdi blob'unu oluştur
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # YOLOv4 modelini kullanarak nesne tespiti yap
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    frame = person_counter.process_frame(frame, outs)

    # Frame'i video dosyasına yaz
    out.write(frame)

    # Ekrana çıktı verin
    cv2.imshow("output", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Video kaynaklarını serbest bırakın
cap.release()
out.release()
cv2.destroyAllWindows()
