import torch
import timm
from PIL import Image
from torchvision import transforms

MODEL = None
try:
    print("Attempting to load State-of-the-Art Vision Transformer (ViT) model...")
    # This uses timm's built-in, robust downloader and cache.
    MODEL = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
    MODEL.eval()
    print("✅ State-of-the-Art ViT model loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load ViT model. AI analysis will be disabled. Reason: {e}")
    MODEL = None

TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def detect_synthesis_artifacts(filepath):
    """
    Uses a ViT to detect AI-generated images. The heuristic: AI images often don't
    fit well into the model's known real-world categories.
    """
    if not MODEL:
        return "ai_model_unavailable"
    try:
        image = Image.open(filepath).convert('RGB')
        image_tensor = TRANSFORMS(image).unsqueeze(0)
        with torch.no_grad():
            output = MODEL(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob = torch.max(probabilities).item()
        # If the model's top prediction is very uncertain, it's likely an AI image.
        if top_prob < 0.10:
            return "ai_generated_image_detected"
    except Exception as e:
        print(f"AI (ViT) analysis failed: {e}")
        return "ai_analysis_failed"
    return None