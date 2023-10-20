from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("./original_frames/frame0.jpg").convert("RGB")
text_prompt = "car"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)