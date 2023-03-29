import cv2
import numpy as np

class PersonCounter:
    def __init__(self):
        self.detected_people = {}
        self.person_id = 0
        self.up_count = 0
        self.down_count = 0
        self.prev_centers = {}
        self.prev_centers_history = {}
        self.history_length = 2

    def determine_direction(self, x, y, w, h, px, py, pw, ph):
        current_center = (x + w / 2, y + h / 2)
        previous_center = (px + pw / 2, py + ph / 2)

        if current_center[1] < previous_center[1] - 3 :
            return "Up"
        elif current_center[1] > previous_center[1] + 3:
            return "Down"
        else:
            return None

    def process_frame(self, frame, outs):
        conf_threshold = 0.1
        nms_threshold = 0.4
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold and class_id == 0:  # Insan
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (center_x, center_y, width, height) = box.astype('int')

                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, int(width), int(height)])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        if len(indices) > 0:
            person_found = [False] * len(self.detected_people)
            for i in indices.flatten():
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                for j, (person_id, (px, py, pw, ph, direction)) in enumerate(self.detected_people.items()):
                    center_x = x + w / 2
                    center_y = y + h / 2
                    if center_x > px and center_x < px + pw and center_y > py and center_y < py + ph:
                        if direction is None:
                            direction = self.determine_direction(x, y, w, h, px, py, pw, ph)
                            if direction == "Up":
                                self.up_count += 1
                            elif direction == "Down":
                                self.down_count += 1
                        self.detected_people[person_id] = (x, y, w, h, direction)
                        person_found[j] = True
                        person_id_found = person_id
                        break

                if not any(person_found):
                    self.person_id += 1
                    self.detected_people[self.person_id] = (x, y, w, h, None)
                    person_found.append(True)
                    person_id_found = self.person_id

            self.detected_people = {k: v for j, (k, v) in enumerate(self.detected_people.items()) if person_found[j]}
            person_found = [False] * len(self.detected_people)

        for person_id, (x, y, w, h, direction) in self.detected_people.items():
            center_x = x + w // 2
            center_y = y + h // 2
        

            # Sınırlayıcı kutuları çizin ve ID'yi yazdırın
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # İnsanın merkezine bir nokta çiz
            cv2.circle(frame, (x + w // 2, y + h // 2), 3, (0, 0, 255), -1)

            # Hareket yönüne göre ok çiz
            if direction == "Up":
                cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y - 20), (0, 255, 255), 2, tipLength=0.4)
            elif direction == "Down":
                cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y + 20), (0, 255, 255), 2, tipLength=0.4)

            if person_id not in self.prev_centers:
                self.prev_centers[person_id] = (center_x, center_y)

            if person_id not in self.prev_centers_history:
                self.prev_centers_history[person_id] = [(center_x, center_y)]

            # Hareketin düzgünleştirilmesi
            self.prev_centers_history[person_id].append((center_x, center_y))
            if len(self.prev_centers_history[person_id]) > self.history_length:
                self.prev_centers_history[person_id].pop(0)

        self.prev_centers[person_id] = (center_x, center_y)


        # Sayaçları ekrana yazdır
        cv2.putText(frame, f"Up count: {self.up_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Down count: {self.down_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



        return frame
