import os
import json
import requests

IMG_DIR = "data/images"
CAPTIONS_FILE = "data/captions.json"
NUM_IMAGES = 50  # How many images to download

os.makedirs(IMG_DIR, exist_ok=True)

captions = {}

for i in range(NUM_IMAGES):
    try:
        resp = requests.get("https://dog.ceo/api/breeds/image/random")
        data = resp.json()
        url = data["message"]
        fname = f"dog_{i+1}.jpg"
        img_path = os.path.join(IMG_DIR, fname)

        # Download image
        img_resp = requests.get(url)
        with open(img_path, "wb") as f:
            f.write(img_resp.content)

        # Generate simple caption from URL
        breed = url.split("/")[-2].replace("-", " ")
        captions[fname] = f"A {breed} dog"

        print(f"Downloaded {fname}: {captions[fname]}")

    except Exception as e:
        print("Failed:", e)

# Save captions.json
with open(CAPTIONS_FILE, "w") as f:
    json.dump(captions, f, indent=2)

print(f"\nSaved {len(captions)} captions in {CAPTIONS_FILE}")
