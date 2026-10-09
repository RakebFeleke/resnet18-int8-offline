
import socket, time, sys
import torch
from PIL import Image
from torchvision import transforms

# Block all network access AFTER imports are done
def no_network(*args, **kwargs):
    raise RuntimeError("Network access is blocked!")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network

model = torch.jit.load("resnet18_int8.pt")
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open(sys.argv[1]).convert("RGB")
x = preprocess(img).unsqueeze(0)

start = time.perf_counter()
with torch.no_grad():
    out = model(x)
ms = (time.perf_counter() - start) * 1000

print("Predicted class index:", out.argmax(dim=1).item())
print("Time:", round(ms, 2), "ms")
print("Worked with network blocked!")
