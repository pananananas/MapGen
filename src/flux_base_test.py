from mflux import Flux1, Config
import random

image_path = "data/maps/map.png"

flux = Flux1.from_alias(
   alias="schnell",  # "schnell" or "dev"
   quantize=4,       # 4 or 8
)

strengths = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]

for strength in strengths:
    seed = random.randint(0, 1000000000)
    image = flux.generate_image(
        seed=seed,
        prompt="fantasy map dnd mountains island clouds green and blue top down view",
        config=Config(
            num_inference_steps=5,  # "schnell" works well with 2-4 steps, "dev" works well with 20-25 steps
            height=1024,
            width=1024,
            init_image_path=image_path,
            init_image_strength=strength,
            guidance=2.0,
        )
    )
    image.save(path=f"data/maps/flux_base/map_{strength}.png")




# mflux-generate-controlnet \
#   --prompt "fantasy map dnd mountains island clouds green and blue top down view" \
#   --model schnell \
#   --steps 4 \
#   --seed 1 \
#   --height 512 \
#   --width 512 \
#   -q 4 \
#   --controlnet-image-path "data/maps/map.png" \
#   --controlnet-strength 0.5 \
#   --controlnet-save-canny