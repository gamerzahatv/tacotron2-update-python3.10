import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        outputs = {}
        
        # Convolution layer C1
        c1 = F.relu(self.conv1(input))
        outputs['conv1'] = c1

        # Subsampling layer S2
        s2 = F.max_pool2d(c1, (2, 2))
        outputs['pool1'] = s2

        # Convolution layer C3
        c3 = F.relu(self.conv2(s2))
        outputs['conv2'] = c3

        # Subsampling layer S4
        s4 = F.max_pool2d(c3, 2)
        outputs['pool2'] = s4
        
        s4 = torch.flatten(s4, 1)

        # Fully connected layers
        f5 = F.relu(self.fc1(s4.view(-1, 16 * 5 * 5)))
        outputs['fc1'] = f5
        f6 = F.relu(self.fc2(f5))
        outputs['fc2'] = f6
        output = self.fc3(f6)
        outputs['output'] = output

        return outputs

# Define transformations to convert image to the format needed by the model
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (if the model expects 1 channel)
    transforms.Resize((32, 32)),                 # Resize to match input dimensions
    transforms.ToTensor(),                        # Convert to tensor format (scales pixel values to [0,1])
])

# Load your image and apply transformations
image_path = r'backup\Untitled.png'
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Initialize network and create a random input tensor
net = Net()
#input = torch.randn(1, 1, 32, 32)  # Batch of 1, 1 channel, 32x32 pixels

# Pass input through the network and get outputs at each layer
with torch.no_grad():
    layer_outputs = net(input_tensor)

# Visualize the output of each layer (convolution and pooling layers)
for layer_name, output in layer_outputs.items():
    if output.dim() == 4:  # Only visualize 2D feature maps (conv/pooling layers)
        num_channels = output.size(1)
        fig, axes = plt.subplots(1, num_channels, figsize=(15, 15))
        fig.suptitle(f"{layer_name} output")
        for i in range(num_channels):
            axes[i].imshow(output[0, i].detach().numpy(), cmap='gray')
            axes[i].axis('off')
        plt.show()
    else:
        print(f"{layer_name} output shape: {output.shape}")
