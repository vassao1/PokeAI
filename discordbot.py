import discord
import asyncio
import os
import dotenv
import aiohttp
import cv2
import numpy as np
import torch
from typing import Optional
from PIL import Image
from ultralytics import YOLO
from torchvision.models import efficientnet_b2
from torchvision import transforms

dotenv.load_dotenv()

TARGET_DESCRIPTION = "Guess the pokémon and type `@Pokétwo#8236 catch <pokémon>` to catch it!"
POKETWO_ID = 716390085896962058
CHANNEL_ID = 926089216658649119
MIN_CONFIDENCE = 0.85
SLEEP_DELAY = 0.4

class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = {'attempts': 0}
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        self.poke_name_map = {
            'toxtricity-amped': 'toxtricity',
            'oricorio-baile': 'oricorio',
            'urshifu-single-strike': 'urshifu',
            'wishiwashi-solo': 'wishiwashi',
            'wormadam-plant': 'wormadam',
            'zygarde-50': 'zygarde',
            'tatsugiri-curly': 'tatsugiri',
            'squawkabilly-green-plumage': 'tapatoes',
            'shaymin-land': 'shaymin',
            'pumpkaboo-average': 'pumpkaboo',
            'oinkologne-male': 'oinkologne',
            'nidoran-m': 'nidoran',
            'nidoran-f': 'nidoran',
            'morpeko-full-belly': 'morpeko',
            'minior-red-meteor': 'minior',
            'mimikyu-disguised': 'mimikyu',
            'meowstic-male': 'meowstic',
            'meloetta-aria': 'meloetta',
            'lycanroc-midday': 'lycanroc',
            'keldeo-ordinary': 'keldeo',
            'indeedee-male': 'indeedee',
            'gourgeist-average': 'gourgeist',
            'giratina-altered': 'giratina',
            'eiscue-ice': 'eiscue',
            'dudunsparce-two-segment': 'dudunsparce',
            'deoxys-normal': 'deoxys',
            'darmanitan-standard': 'darmanitan',
            'basculin-red-striped': 'basculin',
            'basculegion-male': 'basculegion',
            'aegislash-shield': 'aegislash',
            'cramorant': 'uu',
            'espurr': 'psiau',
            'charcadet': 'carbou',
            'klink': 'tic',
            'scatterbug': 'purmel',
            'chinchou': 'lampi',
            'rowlet': 'bauz',
            'magnemite': 'coil',
            'vanillite': 'sorbebe',
            'greavard': 'bochi',
            'kangaskhan': 'garura',
            'whimsicott': 'elfun',
            'vanillish': 'sorboul',
            'trevenant': 'ohrot',
            'gimmighoul': 'mordudor',
            'ghastly': 'ghos',
            'maushold-family-of-four': 'maushold',
            'impidimp': 'beroba',
            'roggenrola': 'dangoro',
            'marshtomp': 'flobio',
            'jangmo-o': 'jarako',
            'mabosstiff': 'dogrino',
            'rolycoly': 'charbi',
            'rookidee': 'meikro'
        }
        
        self.poke_name_block = {
            'mime-jr',
            'landorus-incarnate',
            'thundurus-incarnate',
            'tornadus-incarnate',
            'enamorus-incarnate',
        }
        
        self._init_models()
        self._init_transform()

    def _init_models(self):
        self.yolo_model = YOLO('models/best.pt')
        dataset_dir = 'data/cropset'
        self.class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.effnet = efficientnet_b2(weights=None)
        self.effnet.classifier[1] = torch.nn.Linear(self.effnet.classifier[1].in_features, len(self.class_names))
        self.effnet.load_state_dict(torch.load('models/PokeAI.pth', map_location=self.device))
        self.effnet.eval()
        self.effnet.to(self.device)

    def _init_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    async def on_ready(self):
        self.http_session = aiohttp.ClientSession()
        print(f'🤖 Logged on as {self.user}')
        print(f'🎯 Device: {self.device}')
        print(f'📊 Classes: {len(self.class_names)}')

    async def close(self):
        if self.http_session:
            await self.http_session.close()
        await super().close()

    async def _download_image(self, url: str) -> Optional[np.ndarray]:
        try:
            async with self.http_session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                img_bytes = await resp.read()
                img_arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
                return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    def _crop_pokemon(self, img: np.ndarray) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        results = self.yolo_model(img)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        
        if len(boxes) == 0:
            return None
        
        areas = [(int(b[2]-b[0]) * int(b[3]-b[1])) for b in boxes]
        idx = np.argmax(areas)
        x1, y1, x2, y2 = map(int, boxes[idx][:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        return img[y1:y2, x1:x2]

    def _recognize_pokemon(self, crop: np.ndarray) -> tuple[str, float]:
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        x = self.transform(crop_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.effnet(x)
            probs = torch.nn.functional.softmax(out.squeeze(), dim=0)
            pred = probs.argmax().item()
            conf = probs[pred].item()
            poke_name = self.class_names[pred]
        
        poke_name_lower = poke_name.lower()
        
        if poke_name_lower in self.poke_name_block:
            return None, 0.0
        
        if poke_name_lower in ['nidoran-m', 'nidoran-f']:
            poke_name_final = 'nidoran'
        else:
            poke_name_final = self.poke_name_map.get(poke_name_lower, poke_name)
        
        return poke_name_final, conf

    def _extract_image_url(self, embed) -> Optional[str]:
        if embed.image and embed.image.url:
            return embed.image.url
        if embed.thumbnail and embed.thumbnail.url:
            return embed.thumbnail.url
        if hasattr(embed, 'fields'):
            for field in embed.fields:
                if 'image' in field.name.lower() and 'http' in field.value:
                    return field.value
        return None
        
    async def on_message(self, message):
        if message.author.id != POKETWO_ID:
            return
        
        if message.channel.id != CHANNEL_ID:
            return

        if not message.embeds:
            return

        matched_embeds = [e for e in message.embeds if TARGET_DESCRIPTION in (e.description or "")]
        if not matched_embeds:
            return

        self.stats['attempts'] += 1

        for embed in matched_embeds:
            image_url = self._extract_image_url(embed)
            
            if not image_url:
                print("⚠️ No image found in embed")
                continue

            img = await self._download_image(image_url)
            if img is None:
                print("❌ Failed to download image")
                continue

            crop = self._crop_pokemon(img)
            if crop is None:
                print("❌ Failed to detect pokemon")
                continue

            poke_name, conf = self._recognize_pokemon(crop)
            
            if poke_name is None:
                print(f"🚫 Blocked pokemon")
                continue

            if conf < MIN_CONFIDENCE:
                print(f"⚠️ Low confidence: {conf:.2f} for {poke_name}")
                continue

            await asyncio.sleep(SLEEP_DELAY + np.random.rand())
            await message.channel.send(f"<@{POKETWO_ID}> c {poke_name}")
            print(f"✅ Sent: {poke_name} (conf: {conf:.2f})")
    
client = MyClient()
client.run(os.getenv("DISCORD_TOKEN"))