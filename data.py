import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from xml.etree.ElementTree import parse
import numpy as np

class NotaDataset(Dataset):
    def __init__(self, root, train, transform=None):
        self.root = root
        self.is_train = train

        if self.is_train:
            self.img_dir = '/train/'
        else:
            self.img_dir = '/test/'
        self.transform = transform
        self.img_name_list = os.listdir("{}{}{}".format(self.root, self.img_dir, "img/"))

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self,idx):
        #Image
        img_name = "{}{}{}{}".format(self.root, self.img_dir, "img/", self.img_name_list[idx])
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        if not self.is_train:
            return image

        #Annotation
        annotation_file_name = "{}{}{}{}".format(self.root, self.img_dir, "annotations/", self.img_name_list[idx])
        annotation_file_name = annotation_file_name.replace(".jpg","")
        annotation_file_name = annotation_file_name.replace(".JPG","")
        annotation_file_name += ".xml"

        label_dict = {"neutral": 1, "anger": 2, "surprise": 3, "smile": 4, "sad": 5}
        bndboxes = []
        labels = []

        tree = parse(annotation_file_name)
        tree_root = tree.getroot()
        objects = tree_root.findall("object")

        for obj in objects:
            box = obj.find("bndbox")
            xmin = int(box.findtext("xmin"))
            ymin = int(box.findtext("ymin"))
            xmax = int(box.findtext("xmax"))
            ymax = int(box.findtext("ymax"))
            bndboxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(label_dict[obj.findtext("name")]))

        bndboxes = torch.as_tensor(bndboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        annotations = dict()
        annotations["boxes"] = bndboxes
        annotations["labels"] = labels
        annotations["image_id"] = torch.tensor([idx])

        return image, annotations

def collate_fn(batch):
    return tuple(zip(*batch))
