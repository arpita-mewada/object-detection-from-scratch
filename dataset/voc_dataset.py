import os
import xml.etree.ElementTree as ET
import cv2
import torch
from torch.utils.data import Dataset

CLASS_MAP = {
    "person": 0,
    "car": 1,
    "dog": 2
}

class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, split_file, img_size=224):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.img_size = img_size

        with open(split_file) as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # ---- Load image ----
        img_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape

        # ---- Load annotation ----
        ann_path = os.path.join(self.annotation_dir, image_id + ".xml")
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in CLASS_MAP:
                continue

            label = CLASS_MAP[class_name]
            bndbox = obj.find("bndbox")

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Convert to center format
            x_center = (xmin + xmax) / 2.0 / w
            y_center = (ymin + ymax) / 2.0 / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            boxes.append([x_center, y_center, box_w, box_h])
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # ---- Resize image ----
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target



