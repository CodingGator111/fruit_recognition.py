import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


DATA_DIR = '/home/nvidia/fruit_dataset_split'
MODEL_SAVE_PATH = 'best_fruit_model.pth'
LABELS_SAVE_PATH = 'classes.txt'

BATCH_SIZE = 64        # Decrease to 32 if you encounter Out-Of-Memory errors
NUM_EPOCHS = 10        # Adjust as needed
LEARNING_RATE = 0.001
NUM_WORKERS = 4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    # Standard image transformations for Fruits-360 (100x100)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("--> Loading datasets...")
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), 
                      num_workers=NUM_WORKERS, pin_memory=True)
        for x in ['train', 'val']
    }

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"--> Dataset loaded! Total classes: {num_classes}")


    with open(LABELS_SAVE_PATH, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"--> Class labels saved to '{LABELS_SAVE_PATH}'")

 
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    start_time = time.time()

    print("\nStarting Training Loop...\n" + "=" * 40)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase].samples)
            epoch_acc = running_corrects.double() / len(image_datasets[phase].samples)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Save the best performing weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"--> Saved new best checkpoint with Val Acc: {best_acc:.4f}")

        print("\nTraining completed!")

    total_time = time.time() - start_time
    print("=" * 40)
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()
