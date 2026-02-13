import torch
from torchvision import transforms
from PIL import Image
from model import MNISTNet


def predict(image_path: str) -> int:
    """Predicts the digit in an image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load('mnist/mnist_model.pth', map_location=device))
    model.eval()
    
    # Transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load and process image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = predict(sys.argv[1])
        print(f"Predicted digit: {result}")
    else:
        print("Usage: python predict.py <image_path>")
