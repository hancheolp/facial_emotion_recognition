import torch
import torchvision.transforms as transforms
from data import NotaDataset
import model
from tqdm import tqdm
from PIL import ImageDraw

#Hyper-parameters
num_classes = 6

state_filename = 'checkpoints/epoch-5.pth'
label_dict = {"background": 0, "neutral": 1, "anger": 2, "surprise": 3, "smile": 4, "sad": 5}
index_to_label = {index: label for label, index in label_dict.items()}

#Loading dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = NotaDataset(root="data", train=False, transform=transform)

#Setting a device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Loading our model
model = model.get_model(num_classes)
model.load_state_dict(torch.load(state_filename))
model.eval()

#Moving our model into the device
model.to(device)

#Removing overlapped boxes
def area(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    dx = min(xmax1, xmax2) - max(xmin1, xmin2)
    dy = min(ymax1, ymax2) - max(ymin1, ymin2)
    return dx * dy if (dx >= 0) and (dy >= 0) else 0

def union(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    return min(xmin1, xmin2), min(ymin1, ymin2), max(xmax1, xmax2), max(ymax1, ymax2)

with torch.no_grad():
    print("# Recognizing facial emotions from test images")
    for index, img in enumerate(tqdm(dataset)):
        img_pil = transforms.functional.to_pil_image(img)
        draw = ImageDraw.Draw(img_pil)

        img = img.to(device)
        prediction = model([img])[0]

        boxes = dict()
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            box = tuple(map(int, box))

            for rep_box, predictions in boxes.items():
                if area(rep_box, box) > 0:
                    boxes[rep_box].append({
                        'box': box,
                        'label': label,
                        'score': score,
                    })
                    break
            else:
                boxes[box] = [{
                    'box': box,
                    'label': label,
                    'score': score,
                }]

        for rep_box, predictions in boxes.items():
            predictions = sorted(predictions, key=lambda prediction: prediction['score'])
            best_prediction = predictions[-1]

            best_box = best_prediction['box']
            xmin, ymin, xmax, ymax = best_box
            draw.rectangle(best_box, outline=(0, 255, 0))
            draw.text((xmin + 5, ymin + 5), 'Label: {}'.format(
                index_to_label[int(best_prediction['label'])]
            ), fill=(0, 0, 0))
            #print('Label:', index_to_label[int(best_prediction['label'])])

        img_pil.save('data/test_output/test_img_'+str(index)+".png", 'PNG')