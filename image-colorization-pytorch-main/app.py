import os
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, render_template, send_file
from torchvision import transforms
from io import BytesIO
import numpy as np
# Define your custom U-Net
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = block(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = block(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        return torch.sigmoid(self.final(dec1))

# Flask setup
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load("image-colorization-pytorch-main/model/main.pth", map_location=device))
model.eval()

# Transformation: convert any image to grayscale + resize
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
from flask import Flask, render_template, request, url_for

@app.route("/", methods=["GET", "POST"])
def index():
    output_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(file.stream).convert("L")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)[0].cpu().clamp(0, 1)

            output_image = transforms.ToPILImage()(output)
            output_path = os.path.join("static", "output.png")
            output_image.save(output_path)
            output_url = url_for('static', filename='output.png')  # This is the correct URL for static content
            
    return render_template('index.html', result_img=output_url)


if __name__ == "__main__":
    app.run(debug=True)
