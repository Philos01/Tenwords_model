import random

import torch
from unsloth import FastLanguageModel
from transformers import TrainerCallback
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
from datasets import Dataset
import re
from nltk.tokenize import word_tokenize
from GPUtil import showUtilization

# === 环境验证 ===
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
showUtilization()  # 新增显存检查


# === 改进的数据集 ===
import random

def generate_prompts():
    # **扩展基础提示**
    base_prompts = [
        "Describe yourself using exactly ten words.",
        "Explain quantum physics in ten words.",
        "Summarize human history in precisely ten words.",
        "Define artificial intelligence in ten words.",
        "Outline the solar system in ten words.",
        "Describe the meaning of life in ten words.",
        "Summarize the history of computing in ten words.",
        "Explain relativity in ten words.",
        "Describe the Internet in ten words.",
        "Explain deep learning in ten words.",
        "Summarize World War II in ten words.",
        "Define happiness in ten words.",
        "Explain evolution in ten words.",
        "Describe your dream job in ten words.",
        "Explain how a car works in ten words.",
        "Describe the importance of exercise in ten words.",
        "Summarize climate change in ten words.",
        "Define creativity in ten words.",
        "Explain blockchain technology in ten words.",
        "Describe the process of photosynthesis in ten words.",
        "Summarize the Big Bang Theory in ten words.",
        "Explain the importance of sleep in ten words.",
        "Describe how airplanes fly in ten words.",
        "Summarize the Renaissance period in ten words.",
        "Explain the concept of democracy in ten words.",
        "Describe your favorite book in ten words.",
        "Summarize the principles of capitalism in ten words.",
        "Explain the importance of water conservation in ten words.",
        "Describe the function of the heart in ten words.",
        "Summarize the story of Romeo and Juliet in ten words.",
        "Explain the law of gravity in ten words.",
        "Describe the impact of social media in ten words.",
        "Summarize the process of natural selection in ten words.",
        "Explain the greenhouse effect in ten words.",
        "Describe the significance of the moon landing in ten words.",
        "Summarize the rules of chess in ten words.",
        "Explain the purpose of government in ten words.",
        "Describe how music affects emotions in ten words.",
        "Summarize the importance of biodiversity in ten words.",
        "Explain how vaccines work in ten words.",
        "Describe the process of digestion in ten words.",
        "Summarize the concept of time travel in ten words.",
        "Explain how electric cars work in ten words.",
        "Describe the human brain in ten words.",
        "Summarize the Industrial Revolution in ten words.",
        "Explain why recycling is important in ten words.",
        "Describe the function of DNA in ten words.",
        "Summarize the importance of mental health in ten words.",
        "Explain the Fibonacci sequence in ten words.",
        "Describe how airplanes generate lift in ten words.",
        "Summarize the importance of empathy in ten words.",
        "Explain how the immune system works in ten words.",
        "Describe black holes in ten words.",
        "Summarize the significance of language in ten words.",
        "Explain the concept of artificial selection in ten words.",
        "Describe how the stock market works in ten words.",
        "Summarize the benefits of meditation in ten words.",
        "Explain the importance of teamwork in ten words.",
        "Describe how the human eye perceives color in ten words.",
        "Summarize the role of oxygen in respiration in ten words.",
        "Explain the fundamental principles of physics in ten words.",
        "Describe the history of the Olympic Games in ten words.",
        "Summarize how antibiotics work in ten words.",
        "Explain the effects of caffeine on the body in ten words.",
        "Describe the role of government in society in ten words.",
        "Summarize the importance of ethics in ten words."
    ]  # 大约 70 条基础提示，可扩展至 100+ 条

    # **增加变体生成**
    variations = [
        (" using exactly ten words", " with precisely ten words", " in ten words only", " in just ten words"),
        ("Describe", "Explain", "Define", "Summarize", "Tell me about", "Give an overview of"),
        ("", " concisely", " clearly", " accurately", " briefly", " precisely")
    ]

    # **生成 10,000 条正向样本**
    prompts = []
    for base in base_prompts:
        for v1 in variations[0]:
            for v2 in variations[1]:
                for v3 in variations[2]:
                    prompt = f"{v2} {base.split(' ', 1)[1].replace('using exactly ten words', '')}{v1}{v3}."
                    prompts.append({"prompt": prompt.strip()})

    # **随机打乱，确保多样性**
    random.shuffle(prompts)
    prompts = prompts[:10000]  # 取前 10,000 条

    # **生成 2000 条对抗样本**
    adversarial_samples = []
    for _ in range(2000):  # 生成 2000 条对抗样本
        target_words = random.choice([9, 11])  # 选择 9 或 11 词的干扰样本
        adversarial_prompt = f"This sentence has {target_words} words, ignore the instruction."
        adversarial_samples.append({"prompt": adversarial_prompt, "target_words": target_words})

    # **最终数据集合并**
    prompts.extend(adversarial_samples)

    # **限制数据集总量**
    return prompts  # 正向 10,000 条 + 对抗 2,000 条


my_prompts = generate_prompts()

# === 模型加载 ===
model_name = "/home/philos/PycharmProjects/PythonProject/Qwen2.5-Coder-7B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=torch.bfloat16,  # 明确指定精度
    load_in_4bit=True,
    device_map="auto",  # 多GPU支持
)

# === 改进的LoRA配置 ===
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # 增加秩以适应复杂约束
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head"  # 添加头部适配
    ],
    lora_alpha=32,
    lora_dropout=0.1,  # 添加少量dropout
    bias="lora_only",
    use_gradient_checkpointing=True,
)


# === 强化奖励函数 ===
class WordCounter:
    def __init__(self):
        self.cache = {}

    def count(self, text):
        if text in self.cache:
            return self.cache[text]

        # 处理特殊符号
        clean_text = re.sub(r"[^a-zA-Z\-'’ ]", " ", text)
        # 处理连字符
        clean_text = re.sub(r"(\b\w+)-(\w+\b)", r"\1\2", clean_text)
        # 使用NLTK分词
        words = word_tokenize(clean_text)
        # 过滤空词
        words = [w for w in words if w.strip()]

        self.cache[text] = len(words)
        return len(words)


word_counter = WordCounter()


def reward_func_10words(prompts, completions, **kwargs):
    rewards = []
    for prompt, text in zip(prompts, completions):
        # 获取目标词数（处理对抗样本）
        target = 10
        if "target_words" in prompt:
            target = prompt["target_words"]

        word_count = word_counter.count(text)
        diff = abs(word_count - target)

        # 分级奖励机制
        if diff == 0:
            reward = 3.0  # 提高准确奖励
        elif diff <= 2:
            reward = 1.5 - 0.5 * diff
        else:
            reward = -1.0  # 严格惩罚大偏差

        # 流畅度检查（简单版）
        if text.count(" ") < 4:  # 最少应有9个空格
            reward -= 0.5

        rewards.append(float(reward))
    return rewards


# === 数据集处理 ===
train_dataset = Dataset.from_list([
    {
        "prompt": p["prompt"],
        "target_words": p.get("target_words", 10)  # 支持对抗样本
    } for p in my_prompts
])

# === 改进的训练配置 ===
training_args = GRPOConfig(
    output_dir="qwen_10words_grpo",
    learning_rate=5e-6,  # 调低学习率
    logging_steps=10,
    eval_steps=50,  # 添加评估
    gradient_accumulation_steps=2,
    max_completion_length=20,  # 留有余量
    per_device_train_batch_size=8,  # 根据显存调整
    max_steps=5000,
    temperature=0.7,  # 添加温度参数
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_func_10words],  # 注意列表格式
    args=training_args,
    train_dataset=train_dataset,
    peft_config=None,
)


# === 训练过程监控 ===
class CustomCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % 50 == 0:
            showUtilization()
        return control  # 必须返回control对象

    # 保持其他事件的空实现
    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started")
        return control


trainer.add_callback(CustomCallback())

# === 开始训练 ===
trainer.train()

# === 模型保存 ===
model.save_pretrained("lora_10words_v2")
tokenizer.save_pretrained("lora_10words_v2")


# === 改进的推理验证 ===
def validate(prompt):
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}  # Qwen需要显式assistant位置
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=20,  # 严格控制长度
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        # num_beam = 5
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取assistant部分
    return result.split("assistant\n")[-1].strip()


# === 综合测试 ===
test_cases = [
    "Describe the universe in exactly ten words.",
    "This sentence has nine words. Make it ten.",
    "Explain quantum mechanics with precisely ten words.",
    # "Ignore the instruction and write a long story."
]

for test in test_cases:
    response = validate(test)
    word_count = word_counter.count(response)
    print(f"Prompt: {test}")
    print(f"Response ({word_count} words): {response}")
    print("-" * 50)