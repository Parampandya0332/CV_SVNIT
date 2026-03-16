import os
import glob
import time
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import sys

sys.path.append("./Restormer")
from basicsr.models.archs.restormer_arch import Restormer


device = "cuda"

model = Restormer(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_blocks=[4,6,6,8],
    num_refinement_blocks=4,
    heads=[1,2,4,8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='BiasFree'
)

model.load_state_dict(torch.load("best_model.pth"))

model = model.to(device)

model.eval()


input_dir = "./test_images"
output_dir = "./results"

os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

tile_size = 512
tile_overlap = 32

total_time = 0


for path in image_paths:

    img_name = os.path.basename(path)

    img = Image.open(path).convert("RGB")

    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    _, _, H, W = img_tensor.shape

    output = torch.zeros_like(img_tensor)

    weight = torch.zeros_like(img_tensor)

    start = time.time()

    for y in range(0, H, tile_size - tile_overlap):
        for x in range(0, W, tile_size - tile_overlap):

            y1 = min(y + tile_size, H)
            x1 = min(x + tile_size, W)

            tile = img_tensor[:, :, y:y1, x:x1]

            _, _, h, w = tile.shape

            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8

            tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")

            with torch.no_grad():

                out_tile = model(tile)

            out_tile = out_tile[:, :, :h, :w]

            output[:, :, y:y1, x:x1] += out_tile

            weight[:, :, y:y1, x:x1] += 1

    output /= weight

    output = torch.clamp(output, 0, 1)

    output_img = TF.to_pil_image(output.squeeze().cpu())

    output_img.save(os.path.join(output_dir, img_name))

    total_time += time.time() - start


runtime = total_time / len(image_paths)

print("Runtime per image:", runtime)