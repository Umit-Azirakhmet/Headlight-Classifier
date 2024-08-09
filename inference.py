import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from models import EffnetHeadlightClassifier, ResnetHeadlightClassifier, MobilenetHeadlightClassifier
from torchvision import datasets
from torch.utils.data import DataLoader
from datetime import datetime

def load_model(model_path: str, model_type: str):
    models_dict = {
        "Effnet": EffnetHeadlightClassifier,
        "Mobilenet": MobilenetHeadlightClassifier,
        "Resnet": ResnetHeadlightClassifier
    }
    model = models_dict[model_type]()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    return model

def read_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def log_output(image_path: str, output, class_names):
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    predicted_class = probabilities.argmax().item()
    confidence = probabilities.max().item()
    print(f"Image: {image_path}, Predicted: {class_names[predicted_class]}, Confidence: {confidence:.4f}")

def list_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def main(model_type: str, model_path: str, images_path: list, testing_folder: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, model_type).to(device)
    class_names = ["off", "on"]

    if not testing_folder:
        for image_path in images_path:
            image = read_image(image_path).to(device)
            with torch.no_grad():
                output = model(image)
            log_output(image_path, output, class_names)

    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

        validation_dataset = datasets.ImageFolder(root='./dataset/val', transform=transform)
        test_dataset = datasets.ImageFolder(root='./dataset/test', transform=transform)

        validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        validation_accuracy = evaluate_model(model, validation_loader, device)
        test_accuracy = evaluate_model(model, test_loader, device)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_name = model.__class__.__name__

        results = f"## Model: {model_name}\n"
        results += f"**Date:** {current_time}\n\n"
        results += f"**Validation Accuracy:** {validation_accuracy * 100:.2f}%\n\n"
        results += f"**Test Accuracy:** {test_accuracy * 100:.2f}%\n\n\n\n"
        output_path = './accuracy_results.md'


        with open(output_path, 'a') as f:
            f.write(results)

        print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    model_type = 'Resnet'
    model_path = '/home/umit/Downloads/headlights_classification/checkpoints/resnet_checkpoint.pth'
    data_path = './dataset/test'
    images_path = list_files(data_path)
    main(model_type, model_path, images_path, False)
    main(model_type, model_path, images_path, True)
