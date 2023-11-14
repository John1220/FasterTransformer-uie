# -*- coding: utf-8 -*-
# 导包
from vllm import LLM, SamplingParams
# 定义 输入 prompt
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# 采样温度设置为0.8，原子核采样概率设置为0.95。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# 初始化 vLLM engine
llm = LLM(model="/opt/data02/zhan/UIE/cpt_latest/")
# 使用 llm.generate 生成结果
outputs = llm.generate(prompts, sampling_params)

# Print the outputs. 它将输入提示添加到vLLM引擎的等待队列中，并执行vLLM发动机以生成具有高吞吐量的输出。输出作为RequestOutput对象的列表返回，其中包括所有输出标记。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")