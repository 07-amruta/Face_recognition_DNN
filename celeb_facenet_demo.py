import torch
import numpy as np
import random
from torchvision.datasets import CelebA
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt

DATABASE_SIZE = 20
TOTAL_TEST_IMAGES = 10
MAX_KNOWN = 5
THRESHOLD = 1.0

print("Loading CelebA dataset...")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CelebA(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("Building database...")

database = {}
idx = 0
count = 0

while count < DATABASE_SIZE:
    img_tensor, _ = dataset[idx]
    img = to_pil_image(img_tensor)

    face = mtcnn(img)
    if face is not None:
        emb = resnet(face.unsqueeze(0).to(device))
        database[idx] = emb.detach().cpu().numpy()
        count += 1
    idx += 1

print("Database ready.")

def recognize(img):
    face = mtcnn(img)
    if face is None:
        return "No Face", None

    emb = resnet(face.unsqueeze(0).to(device))
    emb = emb.detach().cpu().numpy()

    best_match = None
    min_dist = float("inf")

    for identity, db_emb in database.items():
        dist = np.linalg.norm(emb - db_emb)
        if dist < min_dist:
            min_dist = dist
            best_match = identity

    if min_dist < THRESHOLD:
        result = f"Matched\nDist={min_dist:.3f}"
    else:
        result = f"Unknown\nDist={min_dist:.3f}"

    return result, min_dist

print("Preparing random test set...")

known_indices = random.sample(list(database.keys()),
                              min(MAX_KNOWN, TOTAL_TEST_IMAGES))

unknown_pool = list(range(DATABASE_SIZE + 50,
                          DATABASE_SIZE + 500))

unknown_needed = TOTAL_TEST_IMAGES - len(known_indices)
unknown_indices = random.sample(unknown_pool, unknown_needed)

test_indices = known_indices + unknown_indices
random.shuffle(test_indices)

print("\nShowing randomized images...")

for i, idx in enumerate(test_indices):

    img_tensor, _ = dataset[idx]
    img = to_pil_image(img_tensor)

    result, dist = recognize(img)

    print(f"Image {i+1} -> {result}")

    plt.imshow(img)
    plt.title(result)
    plt.axis('off')
    plt.show()

print("\nDemo finished.")
