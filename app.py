import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matcher.GFSAM import build_model
import argparse
import PIL.Image as Image
from PIL import ImageDraw

# Arguments parsing
parser = argparse.ArgumentParser(description='GFSAM Pytorch Implementation for One-shot Segmentation')

parser.add_argument('--img-size', type=int, default=1024)
parser.add_argument('--dinov2-size', type=str, default="vit_large")
parser.add_argument('--sam-size', type=str, default="vit_h")
parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
parser.add_argument('--sam-weights', type=str, default="models/sam_vit_h_4b8939.pth")


args = parser.parse_args()
transform = transforms.Compose([
            transforms.Resize(size=(args.img_size, args.img_size)),
            transforms.ToTensor()
        ])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
args.device = device
GFSAM = build_model(args)

def visualize_mask_on_image(image, mask):
    mask = mask.squeeze().cpu().numpy()
    mask = np.where(mask > 0.5, 1, 0)
    # use red mask
    mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask = mask.resize(image.size)
    image = Image.blend(image, mask, alpha=0.5)
    return image


def visualize_points_on_image(image, coords_xy, coords_labels):
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    for idx, (coord, label) in enumerate(zip(coords_xy, coords_labels)):
        x = coord[0] * image.width / args.img_size
        y = coord[1] * image.height / args.img_size
        # draw.point((x, y), fill=(255, 0, 0) if label != 1 else (0, 255, 0))
        draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 0, 0) if label != 1 else (0, 255, 0))
    return image
        

def process(ref_img, ref_mask, target_img):
    ref_img = ref_img.convert('RGB')
    target_img = target_img.convert('RGB')
    ref_img_tensor, target_img_tensor = transform(ref_img), transform(target_img)
    ref_mask_tensor = torch.tensor(np.array(ref_mask))
    ref_mask_tensor = F.interpolate(ref_mask_tensor.unsqueeze(0).unsqueeze(0).float(), ref_img_tensor.size()[-2:], mode='nearest')

    with torch.no_grad():
        GFSAM.clear()
        GFSAM.set_reference(ref_img_tensor.unsqueeze(0).unsqueeze(0).to(device), ref_mask_tensor.to(device))
        GFSAM.set_target(target_img_tensor.unsqueeze(0).to(device))
        pred_mask, point_tuple = GFSAM.predict()

    query_pred = visualize_mask_on_image(target_img, pred_mask)
    coords_xy, coords_labels = point_tuple
    query_points = visualize_points_on_image(query_pred, coords_xy, coords_labels)
    query_pred = query_pred.resize((target_img.width, target_img.height))
    return query_pred, query_points

demo = gr.Interface(
    fn=process, 
    title="Official Demo of 🌉GFSAM", 
    description="<div align='center'> \
        [NeurIPS 2024 Spotlight✨] Bridge the Points: Graph-based Few-shot Segment Anything Semantically \
        </div>", 
    inputs=[gr.Image(label="Reference Image", type="pil"), 
     gr.Image(label="Reference Mask", type="pil", image_mode="L"), 
     gr.Image(label="Target Image", type="pil")], 
    outputs=[gr.Image(label="Prediction"), 
             gr.Image(label="Points")],
    allow_flagging=False
)

demo.launch()