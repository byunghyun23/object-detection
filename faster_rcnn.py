from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess the input image
image_path = 'image.jpg'
image = Image.open(image_path)
image_tensor = F.to_tensor(image)

# Perform inference with the Faster R-CNN model
predictions = model([image_tensor])

# Process the predictions
boxes = predictions[0]['boxes'].tolist()
labels = predictions[0]['labels'].tolist()
scores = predictions[0]['scores'].tolist()
# print(labels)
# for i in scores:
#     print(i)

# Load the COCO class labels and create the label decode
labels_decode = []
with open('coco.names') as f:
    labels_decode = f.readlines()
    labels_decode = [decode.rstrip('\n') for decode in labels_decode]

# Load the original image using OpenCV for visualization
image_origin = cv2.imread(image_path)
image_detection = image_origin.copy()

# Display the bounding boxes, labels, and scores on the original image
for box, label, score in zip(boxes, labels, scores):
    if score > 0.8:
        try:
            name = labels_decode[label - 1]
            box = [int(coord) for coord in box]
            x, y, w, h = box
            cv2.rectangle(image_detection, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(image_detection, f'{name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except IndexError:
            pass

# Display the image with bounding boxes and labels
cv2.putText(image_origin, f'Origin', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(image_detection, f'Detection', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
results = np.hstack((image_origin, image_detection))
cv2.imshow('Object Detection', results)
cv2.waitKey(0)
cv2.destroyAllWindows()