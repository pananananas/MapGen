from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import os

# Wykrycie dostępności GPU lub CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HUGGINGFACE_HUB_CACHE = "/net/tscratch/people/plggintowt/huggingface_cache"

# Ładowanie procesora i modelu
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=HUGGINGFACE_HUB_CACHE)
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    cache_dir=HUGGINGFACE_HUB_CACHE
)
model.to(device)

print("Urządzenie:", device)

# Ścieżka do folderu z obrazami
image_folder = "maps"  # Zmień na swoją ścieżkę
output_folder = "outputs"  # Folder do zapisu wyników
os.makedirs(output_folder, exist_ok=True)

# Lista plików obrazów
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Pętla po obrazach
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)

    # Podaj pytanie/prompt dla obrazu
    prompt = """[INST] <image>
Briefly describe the variety of objects and buildings visible in the image, focusing on diversity and notable features. Keep the description concise.
[/INST]"""

    # Przygotowanie danych wejściowych
    inputs = processor(prompt, image, return_tensors="pt").to(device)

    # Generowanie odpowiedzi
    output = model.generate(**inputs, max_new_tokens=50)

    # Dekodowanie wyniku
    description = processor.decode(output[0], skip_special_tokens=True)
    print(f"Opis obrazu {image_file}:")
    print(description)

    # Zapis wyniku do pliku tekstowego
    output_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_description.txt")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(description)

print("Przetwarzanie zakończone. Opisy zapisane w folderze:", output_folder)




