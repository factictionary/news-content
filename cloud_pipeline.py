"""
Script 2: Cloud Image Generation Pipeline
Run on Google Colab or Kaggle with free GPU
Reads JSON from data/ folder structure, generates images
"""

import json
import os
import sys
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ============================================
# 1. SETUP
# ============================================

def install_dependencies():
    packages = ["diffusers", "transformers", "accelerate", "pillow"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

def is_colab():
    return 'google.colab' in sys.modules

def is_kaggle():
    return 'kaggle' in str(Path.cwd()).lower()

# ============================================
# 2. CLOUD IMAGE GENERATOR
# ============================================

class CloudImageGenerator:
    def __init__(self):
        self.pipe = None
        self.device = None
    
    def setup(self):
        import torch
        from diffusers import ZImagePipeline
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32
            print("⚠️ No GPU, using CPU")
        
        self.pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        self.pipe.to(self.device)
        self.pipe.enable_attention_slicing("auto")
        self.pipe.enable_vae_slicing()
        print("✅ Model ready")
        return self
    
    def create_prompt(self, news_item: Dict) -> str:
        headline = news_item.get('headline', 'news')
        category = news_item.get('category', 'general')
        
        prompts = {
            'tech': f"Cartoon illustration of {headline}. Futuristic tech, funny, colorful, 4k",
            'entertainment': f"Funny comic strip of {headline}. Exaggerated cartoon, meme style",
            'politics': f"Satirical political cartoon of {headline}. Caricature, bold lines",
            'weird': f"Surreal cartoon of {headline}. Absurd, weird but funny",
            'sports': f"Action cartoon of {headline}. Dynamic, exaggerated, comic style"
        }
        return prompts.get(category, f"Cartoon of {headline}, funny, colorful")
    
    def generate_image(self, prompt: str, output_path: Path) -> bool:
        if self.pipe is None:
            return False
        try:
            result = self.pipe(prompt=prompt, height=768, width=768, num_inference_steps=9)
            result.images[0].save(output_path)
            return True
        except:
            return False
    
    def process_json(self, json_source: str, output_dir: str = "output") -> Dict:
        print(f"\n📡 Loading: {json_source}")
        
        # Load JSON
        if json_source.startswith('http'):
            import requests
            response = requests.get(json_source)
            data = response.json()
        else:
            with open(json_source, 'r') as f:
                data = json.load(f)
        
        news_items = data.get("items", [])
        print(f"✅ Loaded {len(news_items)} items")
        
        # Extract date from first imageUrl
        date = datetime.now().strftime('%Y-%m-%d')
        if news_items and 'imageUrl' in news_items[0]:
            match = re.search(r'/images/(\d{4}-\d{2}-\d{2})/', news_items[0]['imageUrl'])
            if match:
                date = match.group(1)
        
        # Setup output
        output_path = Path(output_dir)
        images_dir = output_path / "images" / date
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate images
        print(f"\n🎨 Generating images for {date}...")
        generated = 0
        
        for idx, item in enumerate(news_items, 1):
            item_id = item.get('id', f"{idx:03d}")
            img_path = images_dir / f"{item_id}.jpg"
            
            if img_path.exists():
                print(f"   [{idx}/{len(news_items)}] {item_id} - exists")
                continue
            
            prompt = self.create_prompt(item)
            print(f"   [{idx}/{len(news_items)}] Generating {item_id}...")
            
            if self.generate_image(prompt, img_path):
                generated += 1
                print(f"      ✅ Saved")
            else:
                print(f"      ⚠️ Failed")
        
        # Save final JSON (preserves all original fields including imageUrl)
        final_json = output_path / f"{date}.json"
        with open(final_json, 'w', encoding='utf-8') as f:
            json.dump({"items": news_items}, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Complete! {generated} images generated")
        print(f"   JSON: {final_json}")
        print(f"   Images: {images_dir}")
        
        return {"generated": generated, "total": len(news_items)}


# ============================================
# 3. MAIN
# ============================================

def main():
    print("\n" + "="*60)
    print("🎭 CLOUD IMAGE GENERATION")
    print("="*60)
    
    if is_colab():
        from google.colab import drive
        drive.mount('/content/drive')
        output_dir = "/content/drive/MyDrive/news_output"
    elif is_kaggle():
        output_dir = "/kaggle/working/news_output"
    else:
        output_dir = "./news_output"
    
    json_source = input("\n📁 Enter JSON URL or path: ").strip()
    
    if not json_source:
        print("No source provided")
        return
    
    mode = input("\nGenerate images? (y/n, default: y): ").strip().lower()
    
    if mode == 'n':
        print("Exiting")
        return
    
    install_dependencies()
    generator = CloudImageGenerator()
    generator.setup()
    generator.process_json(json_source, output_dir)


if __name__ == "__main__":
    main()
