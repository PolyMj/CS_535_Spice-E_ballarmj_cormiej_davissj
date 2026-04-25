import torch
import clip
from PIL import Image
import json
import argparse
import os
from torchvision.transforms import ToPILImage
import point_cloud_utils as pcu # `pip install point-cloud-utils`

def load_clip_model(device):
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess

def compute_geodiff(guidance_pc_path, output_pc_path):
    if (not os.path.exists(guidance_pc_path)):
        print(f"ERROR: Guidance model not found: {guidance_pc_path}")
        return None
    if (not os.path.exists(output_pc_path)):
        print(f"ERROR: Guidance model not found: {output_pc_path}")
        return None
    
    gpc = pcu.load_mesh_v(guidance_pc_path)
    opc = pcu.load_mesh_v(output_pc_path)

    return float(pcu.chamfer_distance(gpc, opc)) * 10e-2 # GD scaled by 10^-2 as specified in the paper

@torch.no_grad()
def compute_clip_metrics(image_paths, prompt, target_image_path, model, preprocess, device):
    # 1. Encode Text
    text_tokens = clip.tokenize([prompt]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 2. Encode Rendered Images (Generated Outputs)
    image_features_list = []
    for img_path in image_paths:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        img_feat = model.encode_image(image)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        image_features_list.append(img_feat)
    
    avg_image_features = torch.cat(image_features_list, dim=0).mean(dim=0, keepdim=True)
    avg_image_features /= avg_image_features.norm(dim=-1, keepdim=True)

    # CLIP-Sim calculation
    clip_sim = torch.cosine_similarity(avg_image_features, text_features).item()

    # 3. CLIP-Dir calculation
    clip_dir = 0.0
    if target_image_path and os.path.exists(target_image_path):
        if target_image_path.lower().endswith('.pt'):
            data = torch.load(target_image_path, map_location=device)
            
            if isinstance(data, dict):
                src_tensor = data.get('latent') or data.get('image') or next(iter(data.values()))
            else:
                src_tensor = data

            # Fix: Handle the 1024*1024 flattened spatial tensor
            if src_tensor.numel() == 1048576:
                # Reshape to [1, 1024, 1024] or [1024, 1024]
                src_tensor = src_tensor.view(1024, 1024)
                
                # Convert to a format CLIP can preprocess (PIL Image)
                # We normalize to 0-255 if it's not already
                if src_tensor.max() <= 1.0:
                    src_tensor = src_tensor * 255
                
                src_pil = ToPILImage()(src_tensor.byte().cpu())
                
                # Now ENCODE it to get the 768 dimension
                src_feat = model.encode_image(preprocess(src_pil).unsqueeze(0).to(device))
            
            elif src_tensor.shape[-1] == 768:
                # It's already an embedding
                src_feat = src_tensor
            else:
                # Fallback for other shapes
                src_feat = src_tensor

        # Ensure embedding is 2D [1, D] and normalized
        src_feat = src_feat.view(1, -1).float()
        src_feat /= src_feat.norm(dim=-1, keepdim=True)

        # Dimension Check before subtraction
        if src_feat.shape[1] == avg_image_features.shape[1]:
            dir_image = avg_image_features - src_feat
            dir_text = text_features 
            clip_dir = torch.cosine_similarity(dir_image, dir_text).item()
        else:
            print(f"Error: Dimension mismatch. Source: {src_feat.shape[1]}, Generated: {avg_image_features.shape[1]}")
            clip_dir = 0.0

    return clip_sim, clip_dir


def create_metrics(images_dir:str, prompt:str, guidance:str, result:str, source_image:str, output_json:str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_model(device)

    img_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                   if f.lower().endswith(img_extensions)]

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    sim, direction = compute_clip_metrics(image_files, prompt, source_image, model, preprocess, device)

    # Geometric difference
    gd = None
    if (guidance is not None and result is not None):
        gd = compute_geodiff(guidance, result)

    results = {
        "guidance": os.path.basename(guidance),
        "prompt": prompt,
        "clip_sim": round(sim, 4),
        "clip_dir": round(direction, 4),
        "geometric_difference": round(gd, 4) if gd is not None else "N/A"
    }

    if os.path.dirname(output_json):
        output_path = output_json
    else:
        output_path = os.path.join("metrics", output_json)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    print(f"--- Results ---")
    print(f"CLIP-Sim: {sim:.4f}")
    print(f"CLIP-Dir: {direction:.4f}")
    print(f"GD:       {gd:.4f}")
    
    with open(output_path, "w") as f:
        json.dump(results, indent=4, fp=f)
    
    print(f"Saved to: metrics/{output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images_dir", required=True, help="Path to rendered images")
    parser.add_argument("-p", "--prompt", required=True, help="The text prompt")
    parser.add_argument("-g", "--guidance", required=False, help="The model used for guidance")
    parser.add_argument("-r", "--result", required=False, help="The model produced as output")
    parser.add_argument("-s", "--source_image", help="Source .pt or image file for CLIP-Dir")
    parser.add_argument("-o", "--output_json", default="metrics.json")
    args = parser.parse_args()

    create_metrics(args.images_dir, args.prompt, args.guidance, args.result, args.source_image, args.output_json)

if __name__ == "__main__":
    main()