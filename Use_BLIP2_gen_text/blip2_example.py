# 开发时间 2024/8/2 16:27
# 开发人员:牧良逢
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
#image_path='/mnt/Data/wangshilong/pycode/HAT-pytorch/COCO_val2014_000000317595.jpg'
#image=Image.open(image_path).convert('RGB')

question='Question: {What does this picture depict} Answer:'
inputs = processor(image, question,return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
