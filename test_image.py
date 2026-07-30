import sys
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = 'best_fruit_model.pth'
LABELS_PATH = 'classes.txt'

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if not os.path.exists(LABELS_PATH):
        print(f"Error: Label file '{LABELS_PATH}' not found. Train the model first.")
        return

    with open(LABELS_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    num_classes = len(class_names)

    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)


    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()


    preprocess = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_label = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    print("\n" + "=" * 40)
    print(f"Image: {image_path}")
    print(f"Predicted Fruit: {predicted_label}")
    print(f"Confidence:      {confidence_pct:.2f}%")
    print("=" * 40 + "\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 test_image.py <path_to_image>")
        print("Example: python3 test_image.py /home/nvidia/fruit_dataset_split/test/Apple\\ Braeburn/0_100.jpg")
    else:
        predict_image(sys.argv[1])
