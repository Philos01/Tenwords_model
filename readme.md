# Ten Words Model

我们微调这个模型是出于兴趣，通过`unsloth`库来调用`lora`来微调模型，基准大模型使用的是`Qwen2.5-7B`，并且使用了[DeepSeekMath](https://arxiv.org/abs/2402.03300)中提出的`GRPO`策略来训练。由于`unsloth`使用到了`triton`库，所以`win`系统可能无法有效地运行，推荐`linux`来无痛训练。

## 我的环境
```bash
Ubuntu 20.04
torch 2.6.0
cuda 11.8
python 3.10
```
本文默认已经完成了`Pytorch(Cuda)`的配置。

## Step1: 环境配置
```bash
pip install unsloth 
pip install trl 
pip install nltk 
pip install GPUtil 
pip install re
# 这里可以使用清华源加速
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
对于nltk库可能会报错没有找到punkt，安装后又会提示punk_tab,大家只需要根据报错提示在Python Console中输入以下命令或者新建一个.py文件来运行：
import nltk
nltk.download('punk')
nltk.download('punkt_tab')
我当时是翻墙才能下载，大家如果下载不了有条件可以尝试翻墙下载，我将这段代码写成package_download.py也放在文件夹了
```

## Step2: 数据准备
我们需要喂给大模型一堆问题，并且明确其只能用十个字来回答，并且每个问题需要以`Prompts`开头。

```bash
my_prompts = [
    {"prompt": "Describe yourself using exactly ten words only."},
    # ……
    # ……
    # ……
]
```
我在代码中提供比较简单的`Prompts`生成代码，并且提供了2000条对抗样本，但由于是简单地排列组合，因此还是略显单调，大家可以根据实际需要多样性地提供`Prompt`。

## Step3: 模型训练
这一步我们使用`lora`微调器来微调模型，基本每个博主讲到大模型都会使用到`lora`，这里就不详讲了。

我们使用的模型是`Qwen2.5-Coder-7B-Instruct`，我是本地下载然后导入的：[Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/main)

```bash
from unsloth import FastLanguageModel

model_name = "/home/philos/PycharmProjects/PythonProject/Qwen2.5-Coder-7B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=torch.bfloat16, 
    load_in_4bit=True,
    device_map="auto"
)
```

大家在运行代码时记得更改文件路径，运行会生成`lora_10words_v2`文件夹，大家可以根据`test.py`代码来测试：

```bash
import torch
from unsloth import FastLanguageModel

model_10words, tokenizer_10words = FastLanguageModel.from_pretrained(
    "lora_10words_v2",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
)
FastLanguageModel.for_inference(model_10words)

prompt = "Describe the universe in exactly ten words."
inputs = tokenizer_10words.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model_10words.generate(
    input_ids=inputs,
    max_new_tokens=20,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer_10words.eos_token_id,
)

print(tokenizer_10words.decode(outputs[0]))
```

### 示例输出
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Describe the universe in exactly ten words.<|im_end|>
<|im_start|>assistant
The universe is vast, containing stars, planets, and dark matter.<|im_end|>
```

大家可以根据自己的想象力来让模型做很多奇奇怪怪的事情，这只是其中一种，我们只需要写好奖励函数，提供对应的数据集。

此代码供大家学习，可以试试各个量级的模型，尝试各种任务，比如回答风格的改变等等~