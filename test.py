# 假设已经训练并保存了LoRa
import torch
from unsloth import FastLanguageModel

model_10words, tokenizer_10words = FastLanguageModel.from_pretrained(
    "lora_10words_v2",
    max_seq_length=2048,
    dtype=torch.bfloat16,  # 明确指定精度
    load_in_4bit=True,
    device_map="auto",
)
FastLanguageModel.for_inference(model_10words)

prompt = "Describe the universe in exactly ten words."
inputs = tokenizer_10words.apply_chat_template(
    [{"role":"user","content":prompt}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model_10words.generate(
    input_ids=inputs,
    max_new_tokens=20,  # 严格控制长度
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer_10words.eos_token_id,
)

print(tokenizer_10words.decode(outputs[0]))
