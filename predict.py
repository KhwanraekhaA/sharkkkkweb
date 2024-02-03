from ultralytics import YOLO
import cv2

class SHARK_DETECTION:
    def __init__(self, model_path):
        # นำโมเดลเข้ามา
        self.model = YOLO(model_path)

    def __call__(self, input_image, output_path):
        # อ่านรูปที่ต้องการที่จะ detect
        img = cv2.imread(input_image)
        # detect เเละเก็บผลลัพธ์ไว้ที่ตัวแปร results
        results = self.model(input_image)[0]

        # ถ้ามีสายพันธุ์มากกว่า 1 ให้แสดงทั้งหมด
        for i in range(len(results.boxes.data)):
            # นำค่าพิกัด bounding box, ค่าความมั่นใจ(confidence ratio), class 
            # มาเก็บไว้ที่ตัวแปร boxes
            boxes = results.boxes.data[i].numpy().tolist()
            # ย่อขนาดของ bounding box
            new_boxes = [int(coord) for coord in boxes[:4]]
            box_width = new_boxes[2] - new_boxes[0]
            box_height = new_boxes[3] - new_boxes[1]
            new_width = int(box_width / 2)
            new_height = int(box_height / 2)
            new_boxes[0] += new_width
            new_boxes[1] += new_height
            new_boxes[2] -= new_width
            new_boxes[3] -= new_height

            # สร้าง bounding box
            cv2.rectangle(img, (new_boxes[0], new_boxes[1]),
                          (new_boxes[2], new_boxes[3]), [0, 255, 0], 2)

            # เพิ่ม Text ที่บอก class เเละ confident ratio
            text = f'{results.names[int(boxes[5])]}:{int(boxes[4]*100)}%'
            text_position = (new_boxes[0], new_boxes[3] + (i + 1) * 20)    # เพิ่มขึ้นตามลำดับ

            cv2.putText(img,
                        text,
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        [225, 0, 0],
                        thickness=2)

        # บันทึกภาพลงที่ output_path
        cv2.imwrite(output_path, img)
