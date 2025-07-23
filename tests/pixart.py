import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "d:\\models\\pixart-xl-2-1024-ms",
    torch_dtype=torch.bfloat16,
).to("cuda")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
prompt = "Tiger in a future cyber city, colors that match the jungle, detailed, 8k"
image = pipeline(prompt).images[0]
image.save("test.webp")