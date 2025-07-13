import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import numpy as np
import os
from skimage.color import rgb2hsv
import io
import base64
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import Request as StarletteRequest
import asyncio
from skimage.filters import sobel
from skimage.feature import canny

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 256

vgg = vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
postprocess = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage(),
])

def get_features(image, model, layer_indices):
    features = {}
    x = image
    for idx, layer in enumerate(model):
        x = layer(x)
        if idx in layer_indices:
            features[idx] = x
    return features

content_layers = [12]
style_layers = [0, 2, 5, 10, 19]

def compute_gram_matrix(feature):
    C = feature.size(1)
    H = feature.size(2)
    W = feature.size(3)
    feature = feature.view(C, H * W)
    gram = torch.matmul(feature, feature.t()) / (C * H * W)
    return gram

def load_styles_from_folder(folder_path="styles/"):
    style_images = []
    style_gram_matrices = []
    style_names = []
    artist_styles = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                artist_name = os.path.splitext(filename)[0].split("-")[0].strip().lower()
                if artist_name not in artist_styles:
                    artist_styles[artist_name] = []
                artist_styles[artist_name].append(img_tensor)
            except Exception as e:
                print(f"Error when loading {filename}: {e}")

    for artist_name, tensors in artist_styles.items():
        gram_matrices = []
        for tensor in tensors:
            features = get_features(tensor, vgg, style_layers)
            layer_grams = [compute_gram_matrix(features[layer]) for layer in style_layers]
            gram_matrices.append(layer_grams)

        avg_gram_matrices = []
        for layer_idx in range(len(style_layers)):
            layer_grams = torch.stack([grams[layer_idx] for grams in gram_matrices], dim=0)
            avg_gram = torch.mean(layer_grams, dim=0)
            avg_gram_matrices.append(avg_gram)

        style_gram_matrices.append(avg_gram_matrices)
        style_names.append(artist_name)
        style_images.append(tensors[0])

    return style_images, style_gram_matrices, style_names

style_images, style_gram_matrices, style_names = load_styles_from_folder()

if not style_gram_matrices:
    raise ValueError("No styles found in the specified folder. Please ensure the 'styles/' directory contains valid image files.")

print(f"Loaded styles : {style_names}")

def calculate_loss(gen_image, content_image, style_grams):
    gen_features = get_features(gen_image, vgg, content_layers + style_layers)
    content_features = get_features(content_image, vgg, content_layers)

    content_loss = 0.0
    style_loss = 0.0

    target = content_features[12].detach()
    content_loss = torch.mean((gen_features[12] - target) ** 2)

    for layer, target_gram in zip(style_layers, style_grams):
        gen_feature = gen_features[layer]
        gen_gram = compute_gram_matrix(gen_feature)
        style_loss += torch.mean((gen_gram - target_gram) ** 2)

    return content_loss, style_loss

async def style_transfer(content_image, style_grams, request: Request, num_steps=150):
    content_image = content_image.clone().requires_grad_(True).to(device)
    optimizer = optim.LBFGS([content_image])
    last_iteration = 0

    for i in range(num_steps):
        if await request.is_disconnected():
            print(f"Request canceled at epoc {i + 1}/{num_steps}.")
            return content_image.clamp(-1, 1), i + 1

        def closure():
            optimizer.zero_grad()
            content_loss, style_loss = calculate_loss(content_image, content_image, style_grams)
            total_loss = content_loss + 500 * style_loss
            total_loss.backward()
            return total_loss
        optimizer.step(closure)

        last_iteration = i + 1
        if (i + 1) % 50 == 0:
            print(f"Epocs {i + 1}/{num_steps}")

    return content_image.clamp(-1, 1), last_iteration

def analyze_image(image):
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    mean_color = img_np.mean(axis=(0, 1))
    brightness = np.mean(mean_color)
    variance = np.var(img_np)
    hsv_img = rgb2hsv(img_np)
    saturation = np.mean(hsv_img[:, :, 1])
    gradient_mean = np.mean([np.abs(np.gradient(img_np[:, :, i])) for i in range(3)])

    edges = canny(np.mean(img_np, axis=2), sigma=2)
    edge_density = edges.sum() / (edges.shape[0] * edges.shape[1])

    sobel_edges = sobel(np.mean(img_np, axis=2))
    edge_contrast = np.std(sobel_edges)

    img_tensor = preprocess(Image.fromarray((img_np * 255).astype(np.uint8))).unsqueeze(0).to(device)
    features = get_features(img_tensor, vgg, style_layers)
    texture_complexity = features[0].var().item()
    dominant_color = np.argmax(mean_color)

    return brightness, variance, texture_complexity, dominant_color, saturation, gradient_mean, edge_density, edge_contrast

def choose_style(image):
    (brightness, variance, texture_complexity, dominant_color,
     saturation, gradient_mean, edge_density, edge_contrast) = analyze_image(image)

    print(f"Analyse - Brightness: {brightness:.3f}, Variance: {variance:.3f}, "
          f"Complexity: {texture_complexity:.3f}, Dominant Color: {['Rouge','Vert','Bleu'][dominant_color]}, "
          f"Saturation: {saturation:.3f}, Gradient: {gradient_mean:.3f}, "
          f"Edge density: {edge_density:.3f}, Edge contrast: {edge_contrast:.3f}")

    style_scores = {style: 0 for style in style_names}

    for style in style_scores:
        if style == "vincent van gogh":
            if brightness < 0.5 and saturation > 0.4:
                style_scores[style] += 3
            if variance > 0.05 or edge_contrast > 0.1:
                style_scores[style] += 2
            if texture_complexity > 1000:
                style_scores[style] += 1

        elif style == "edvard munch":
            if brightness < 0.5 and saturation < 0.4:
                style_scores[style] += 2
            if gradient_mean < 0.02:
                style_scores[style] += 1
            if edge_density < 0.05:
                style_scores[style] += 1

        elif style == "katsushika hokusai":
            if gradient_mean > 0.1 and saturation < 0.3:
                style_scores[style] += 3
            if edge_density > 0.1:
                style_scores[style] += 2
            if brightness > 0.5:
                style_scores[style] += 1

        elif style == "claude monet":
            if brightness > 0.45 and texture_complexity > 800:
                style_scores[style] += 4
            if variance > 0.06:
                style_scores[style] += 2
            if saturation > 0.5:
                style_scores[style] += 1

        elif style == "kandinsky":
            if variance > 0.1 or saturation > 0.6:
                style_scores[style] += 3
            if edge_contrast > 0.2:
                style_scores[style] += 2
            if dominant_color == 0:
                style_scores[style] += 1

    for style, score in style_scores.items():
        print(f"Style: {style}, Score: {score}")

    max_score = max(style_scores.values())
    top_styles = [style for style, score in style_scores.items() if score == max_score]

    if len(top_styles) == 1:
        best_style = top_styles[0]
    else:
        if brightness > 0.5:
            luminosity_friendly = ["claude monet", "katsushika hokusai"]
            for candidate in top_styles:
                if candidate in luminosity_friendly:
                    best_style = candidate
                    break
            else:
                best_style = top_styles[0]
        else:
            dark_friendly = ["vincent van gogh", "edvard munch", "kandinsky"]
            for candidate in top_styles:
                if candidate in dark_friendly:
                    best_style = candidate
                    break
            else:
                best_style = top_styles[0]

    return style_names.index(best_style)

class ImageRequest(BaseModel):
    image_data: str

@app.post("/transform-image")
async def transform_image(request: ImageRequest, starlette_request: StarletteRequest):
    try:
        image_data = request.image_data
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        image_tensor = preprocess(input_image).unsqueeze(0).to(device)

        style_idx = choose_style(image_tensor.squeeze(0))
        if style_idx >= len(style_gram_matrices):
            style_idx = 0
        selected_style = style_names[style_idx]
        print(f"Style choisi : {selected_style}")

        styled_image, iteration = await style_transfer(image_tensor, style_gram_matrices[style_idx], starlette_request)
        styled_image = styled_image.squeeze(0).cpu()
        styled_image = postprocess(styled_image)

        buffered = io.BytesIO()
        styled_image.save(buffered, format="PNG")
        styled_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        styled_image_base64 = f"data:image/png;base64,{styled_image_base64}"

        return JSONResponse(content={
            "transformed_image": styled_image_base64,
            "selected_style": selected_style,
            "status": "completed",
            "iteration": iteration
        })
    except RuntimeError as e:
        print(f"Error : {str(e)}")
        styled_image = styled_image.squeeze(0).cpu() if 'styled_image' in locals() else image_tensor.squeeze(0).cpu()
        styled_image = postprocess(styled_image)

        buffered = io.BytesIO()
        styled_image.save(buffered, format="PNG")
        styled_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        styled_image_base64 = f"data:image/png;base64,{styled_image_base64}"

        return JSONResponse(content={
            "transformed_image": styled_image_base64,
            "selected_style": selected_style if 'selected_style' in locals() else "unknown",
            "status": "interrupted",
            "iteration": iteration if 'iteration' in locals() else 0,
            "message": f"Transfer cancel at iteration {iteration if 'iteration' in locals() else 'inconnue'}."
        })
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=1200)