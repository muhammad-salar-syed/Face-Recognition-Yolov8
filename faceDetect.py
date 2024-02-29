import cv2
import mediapipe as mp
import cvzone


class FaceDetector:

    def __init__(self, minDetectionCon=0.5, modelSelection=0):

        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.minDetectionCon,
                                                                model_selection=self.modelSelection)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                if detection.score[0] > self.minDetectionCon:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                    cx, cy = bbox[0] + (bbox[2] // 2), \
                             bbox[1] + (bbox[3] // 2)
                    bboxInfo = {"id": id, "bbox": bbox, "score": detection.score, "center": (cx, cy)}
                    bboxs.append(bboxInfo)
                    if draw:
                        img = cv2.rectangle(img, bbox, (255, 0, 255), 2)

                        cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                    (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 255), 2)
        return img, bboxs


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    while True:

        success, img = cap.read()
        img, bboxs = detector.findFaces(img, draw=False)

        # Check if any face is detected
        if bboxs:
            for bbox in bboxs:
                # bbox contains 'id', 'bbox', 'score', 'center'
                center = bbox["center"]
                x, y, w, h = bbox['bbox']
                score = int(bbox['score'][0] * 100)

                cv2.circle(img, center, 3, (0, 0, 255), cv2.FILLED)
                cv2.putText(img,f'{score}%',(x, y - 10), cv2.FONT_HERSHEY_PLAIN,2, (0, 255, 0), 2)
                cvzone.cornerRect(img, (x, y, w, h))

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()