# FastChat

| [**Demo**](https://chat.lmsys.org/) | [**Discord**](https://discord.gg/HSWAKCrnFx) | [**Twitter**](https://twitter.com/lmsysorg) |

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. The core features include:

- The weights, training code, and evaluation code for state-of-the-art models (e.g., Vicuna).
- A distributed multi-model serving system with web UI and OpenAI-compatible RESTful APIs.

## News

- [2023/08] 🔥 We released **Vicuna v1.5** based on Llama 2 with 4K and 16K context lengths. Download [weights](#vicuna-weights).
- [2023/08] 🔥 We released **LongChat v1.5** based on Llama 2 with 32K context lengths. Download [weights](#longchat).
- [2023/07] We released **Chatbot Arena Conversations**, a dataset containing 33k conversations with human preferences. Download it [here](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations).

<details><summary>More</summary>

- [2023/06] We introduced **MT-bench**, a challenging multi-turn question set for evaluating chatbots. Check out the blog [post](https://lmsys.org/blog/2023-06-22-leaderboard/).
- [2023/06] We introduced **LongChat**, our long-context chatbots and evaluation tools. Check out the blog [post](https://lmsys.org/blog/2023-06-29-longchat/).
- [2023/05] We introduced **Chatbot Arena** for battles among LLMs. Check out the blog [post](https://lmsys.org/blog/2023-05-03-arena).
- [2023/03] We released **Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality**. Check out the blog [post](https://vicuna.lmsys.org).

</details>

<a href="https://chat.lmsys.org"><img src="assets/demo_narrow.gif" width="70%"></a>

## Contents

- [Install](#install)
- [Model Weights](#model-weights)
- [Inference with Command Line Interface](#inference-with-command-line-interface)
- [Serving with Web GUI](#serving-with-web-gui)
- [API](#api)
- [Evaluation](#evaluation)
- [Fine-tuning](#fine-tuning)
- [Citation](#citation)

## Install

### Method 1: With pip

```bash
pip3 install "fschat[model_worker,webui]"
```

### Method 2: From source

1. Clone this repository and navigate to the FastChat folder

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
```

If you are running on Mac:

```bash
brew install rust cmake
```

2. Install Package

```bash
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"
```

## Model Weights

### Vicuna Weights

[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) is based on LLaMA and should be used under LLaMA's [model license](https://github.com/facebookresearch/llama/blob/main/LICENSE).

You can use the commands below to start chatting. It will automatically download the weights from Hugging Face repos.
See more command options and how to handle out-of-memory in the "Inference with Command Line Interface" section below.

**NOTE: `transformers>=4.31` is required for 16K versions.**

| Size | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| 7B   | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5`  | [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)   |
| 7B-16k   | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5-16k`  | [lmsys/vicuna-7b-v1.5-16k](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k)   |
| 13B  | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-13b-v1.5` | [lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) |
| 13B-16k  | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-13b-v1.5-16k` | [lmsys/vicuna-13b-v1.5-16k](https://huggingface.co/lmsys/vicuna-13b-v1.5-16k) |
| 33B  | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-33b-v1.3` | [lmsys/vicuna-33b-v1.3](https://huggingface.co/lmsys/vicuna-33b-v1.3) |

**Old weights**: see [docs/vicuna_weights_version.md](docs/vicuna_weights_version.md) for all versions of weights and their differences.

### LongChat

We release [LongChat](https://lmsys.org/blog/2023-06-29-longchat/) models under LLaMA's [model license](https://github.com/facebookresearch/llama/blob/main/LICENSE).

| Size | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| 7B   | `python3 -m fastchat.serve.cli --model-path lmsys/longchat-7b-32k-v1.5`  | [lmsys/longchat-7b-32k](https://huggingface.co/lmsys/longchat-7b-32k-v1.5)   |

### FastChat-T5

You can use the commands below to chat with FastChat-T5. It will automatically download the weights from Hugging Face repos.

| Size | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| 3B   | `python3 -m fastchat.serve.cli --model-path lmsys/fastchat-t5-3b-v1.0`  | [lmsys/fastchat-t5-3b-v1.0](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) |

## Inference with Command Line Interface

<a href="https://chat.lmsys.org"><img src="assets/screenshot_cli.png" width="70%"></a>

(Experimental Feature: You can specify `--style rich` to enable rich text output and better text streaming quality for some non-ASCII content. This may not work properly on certain terminals.)

#### Supported Models

FastChat supports a wide range of models, including
LLama 2, Vicuna, Alpaca, Baize, ChatGLM, Dolly, Falcon, FastChat-T5, GPT4ALL, Guanaco, MTP, OpenAssistant, RedPajama, StableLM, WizardLM, and more.

See a complete list of supported models and instructions to add a new model [here](docs/model_support.md).

#### Single GPU

The command below requires around 14GB of GPU memory for Vicuna-7B and 28GB of GPU memory for Vicuna-13B.
See the ["Not Enough Memory" section](#not-enough-memory) below if you do not have enough memory.
`--model-path` can be a local folder or a Hugging Face repo name.

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3
```

#### Multiple GPUs

You can use model parallelism to aggregate GPU memory from multiple GPUs on the same machine.

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --num-gpus 2
```

Tips:
Sometimes the "auto" device mapping strategy in huggingface/transformers does not perfectly balance the memory allocation across multiple GPUs.
You can use `--max-gpu-memory` to specify the maximum memory per GPU for storing model weights.
This allows it to allocate more memory for activations, so you can use longer context lengths or larger batch sizes. For example,

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --num-gpus 2 --max-gpu-memory 8GiB
```

#### CPU Only

This runs on the CPU only and does not require GPU. It requires around 30GB of CPU memory for Vicuna-7B and around 60GB of CPU memory for Vicuna-13B.

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --device cpu
```

#### Serving Fine-tuned Peft Model on CPU

If you have a [peft model](https://github.com/huggingface/peft) fine-tuned from base model (e.g. Llama2 etc.) using [BigDL Distributed fine-tuning](https://github.com/intel-analytics/BigDL/tree/main/docker/llm/finetune/lora#run-bf16-optimized-lora-finetuning-on-kubernetes-with-oneccl), the fine-tuned large model, peft model, can be served on FastChat with standard APIs with the help of CPU. The following presents how to quickly start FastChat serving of peft model inference.

Fisrt, you should prepare available base and peft models. If you have finished the BigDL fine-tuning mentioned before, you will find a peft model output on your NFS, which contains the incremental weights representing the new knowledge to the base model learned during fine-tuing process. In addition, you need to make sure the `base_model_name_or_path` in config file under peft model directory, which refers to the remote or local URL of base model file, has been set correctly. Namely, the base model should be available on local storage or accessible HuggingFace repo.

For a quick start or users who wants to serve only an open model, you can skip the fine-tuning process by downloading other existing peft models directly, and BigDL supports any standard peft model on FastChat as well.

Then, the inference can be easily invoked by FastChat CLI (please make sure you have installed it correctly):

```bash
python -m fastchat.serve.cli --model-path <your_peft_model_path_or_repo_name> --device cpu
```

**Note: To active peft-model matching, the `model-path` parameter must have `peft` in the path.**

After that, you can have a conversion with the fine-tuned model and it is able to give a better answer than the base model.:

```bash
# base model

Human: it is 2023 now, so who is the current American president
Assistant: Here's a list of all the US Presidents since the foundation of the country.
1. George Washington (1789-1797)
2. John Adams (1797 - 1801)
3. Thomas Jefferson (1801-1809)
4. James Madison (1809-1817)
5. James Monroe (1817-1825)
6. John Quincy Adams (1825-1829)
7. Andrew Jackson (1829-1837)
8. Martin Van Buren (1837-1841)
9. William Henry Harrison (1841)
10. John Tyler (1841-1845)
11. James K. Polk (1845-1849)
12. Zachary Taylor (1849-1850)
13. Millard Fillmore (1850-1853)
14. Franklin Pierce (1853-1857)
15. James Buchanan (1857-1861)
16. Abraham Lincoln (1861-1865)
17. Andrew Johnson (1865-1869)
18. Ulysses S. Grant (1869-1877)
19. Rutherford B. Hayes (1877 - 1881)
20. James A. Garfield (1881)
21. Chester A. Arthur (1881-1885)
22. Grover Cleveland (1885-1889)
23. Benjamin Harrison (1889-1893)
24. Witiiam McKinley (1897 -1901)
25. Theodore Roosevelt (1901-1909)
26. William Howard Taft (1909-1913)
27. Woodrow Wilson (1913-1921)
28. Warren G. Harding
```

```bash
# peft model after fine-tuning
Human: it is 2023 now, so who is the current American president
Assistant: The current American president is Joe Biden. He is the 49th president of the United States.
```

Moreover, according to our experience, FastChat can have better performance with optimizations of [BigDL Nano](https://github.com/intel-analytics/BigDL/tree/main#nano) etc., so you can improve the inference like below:

```bash
pip install --pre --upgrade bigdl-nano # install BigDL Nan
o
export OMP_NUM_THREADS=... # value of physical cores on one socket as your CPU platform
export CPU_SET=0-47 # this is the set of cpu core num to use by FastChat, it should have the same size as OMP_NUM_THREADS, e.g. 0-47 will bind core 0 to core 47, 48 cores totally
numactl -C $CPU_SET -m 0 python -m fastchat.serve.cli --model-path <peft_model_path_or_repo_name> --device cpu
```

##### Some possible issues and solutions

1. `RuntimeError: The size of tensor a (8912896) must match the size of tensor b (4096) at non-singleton dimension 1`
Need to check `base_model_name_or_path` in config of the peft model, which should not match `bigdl`.

2. `RecursionError: maximum recursion depth exceeded while getting the str of an object.`
It may be related to the version unmatched between tokenizer files in base model and `transformer` lib (refer to [this issue](https://github.com/huggingface/transformers/issues/22762)). You can get the updated tokenizer from [here](https://huggingface.co/huggyllama/llama-7b) and replace these files in base model directory.

Use Intel AI Accelerator AVX512_BF16/AMX to accelerate CPU inference.

```
CPU_ISA=amx python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --device cpu
```

#### Metal Backend (Mac Computers with Apple Silicon or AMD GPUs)

Use `--device mps` to enable GPU acceleration on Mac computers (requires torch >= 2.0).
Use `--load-8bit` to turn on 8-bit compression.

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --device mps --load-8bit
```

Vicuna-7B can run on a 32GB M1 Macbook with 1 - 2 words / second.

#### Intel XPU (Intel Data Center and Arc A-Series GPUs)

Install the [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html). Set the OneAPI environment variables:

```
source /opt/intel/oneapi/setvars.sh
```

Use `--device xpu` to enable XPU/GPU acceleration.

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --device xpu
```

Vicuna-7B can run on an Intel Arc A770 16GB.

#### Not Enough Memory

If you do not have enough memory, you can enable 8-bit compression by adding `--load-8bit` to commands above.
This can reduce memory usage by around half with slightly degraded model quality.
It is compatible with the CPU, GPU, and Metal backend.

Vicuna-13B with 8-bit compression can run on a single GPU with 16 GB of VRAM, like an Nvidia RTX 3090, RTX 4080, T4, V100 (16GB), or an AMD RX 6800 XT.

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3 --load-8bit
```

In addition to that, you can add `--cpu-offloading` to commands above to offload weights that don't fit on your GPU onto the CPU memory.
This requires 8-bit compression to be enabled and the bitsandbytes package to be installed, which is only available on linux operating systems.

#### More Platforms and Quantization

- For AMD GPU users, please install ROCm and [the ROCm version of PyTorch](https://pytorch.org/get-started/locally/) before you install FastChat. See also this [post](https://github.com/lm-sys/FastChat/issues/104#issuecomment-1613791563).
- FastChat supports GPTQ 4bit inference with [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa). See [docs/gptq.md](/docs/gptq.md).
- FastChat supports AWQ 4bit inference with [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq). See [docs/awq.md](/docs/awq.md).
- [MLC LLM](https://mlc.ai/mlc-llm/), backed by [TVM Unity](https://github.com/apache/tvm/tree/unity) compiler, deploys Vicuna natively on phones, consumer-class GPUs and web browsers via Vulkan, Metal, CUDA and WebGPU.

## Serving with Web GUI

<a href="https://chat.lmsys.org"><img src="assets/screenshot_gui.png" width="70%"></a>

To serve using the web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the webserver and model workers. You can learn more about the architecture [here](docs/server_arch.md).

Here are the commands to follow in your terminal:

#### Launch the controller

```bash
python3 -m fastchat.serve.controller
```

This controller manages the distributed workers.

#### Launch the model worker(s)

```bash
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller .

If you want to use document QA feature, you should run the model-names include `text-embedding-ada-002` and `gpt-3.5-turbo`, because we use set the default embedding model as `text-embedding-ada-002` and set the default qa llm as `gpt-3.5-turbo`

```bash
python3 -m fastchat.serve.model_worker --model-names "Llama-2-7b-chat-hf,text-embedding-ada-002,gpt-3.5-turbo" --model-path lmsys/vicuna-7b-v1.3
```

To ensure that your model worker is connected to your controller properly, send a test message using the following command:

```bash
python3 -m fastchat.serve.test_message --model-name vicuna-7b-v1.3
```

You will see a short output.

#### Launch the OpenAI api server (Optional)

If you want to use document QA feature, you should run OpenAI api server becasue LangChain uses OpenAI model names by default.
Ensure you have assign faux `gpt-3.5-turbo` and `text-embedding-ada-002` model names to our local model, then:

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

#### Launch the Gradio web server

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI. You can open your browser and chat with a model now.
If the models do not show up, try to reboot the gradio web server.

#### (Optional): Advanced Features

- You can register multiple model workers to a single controller, which can be used for serving a single model with higher throughput or serving multiple models at the same time. When doing so, please allocate different GPUs and ports for different model workers.

```
# worker 0
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
# worker 1
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path lmsys/fastchat-t5-3b-v1.0 --controller http://localhost:21001 --port 31001 --worker http://localhost:31001
```

- You can also launch a multi-tab gradio server, which includes the Chatbot Arena tabs.

```bash
python3 -m fastchat.serve.gradio_web_server_multi
```

## API

### OpenAI-Compatible RESTful APIs & SDK

FastChat provides OpenAI-compatible APIs for its supported models, so you can use FastChat as a local drop-in replacement for OpenAI APIs.
The FastChat server is compatible with both [openai-python](https://github.com/openai/openai-python) library and cURL commands.
See [docs/openai_api.md](docs/openai_api.md).

### Hugging Face Generation APIs

See [fastchat/serve/huggingface_api.py](fastchat/serve/huggingface_api.py).

### LangChain Integration

See [docs/langchain_integration](docs/langchain_integration.md).

## Evaluation

We use MT-bench, a set of challenging multi-turn open-ended questions to evaluate models.
To automate the evaluation process, we prompt strong LLMs like GPT-4 to act as judges and assess the quality of the models' responses.
See instructions for running MT-bench at [fastchat/llm_judge](fastchat/llm_judge).

MT-bench is the new recommended way to benchmark your models. If you are still looking for the old 80 questions used in the vicuna blog post, please go to [vicuna-blog-eval](https://github.com/lm-sys/vicuna-blog-eval).

## Fine-tuning

### Data

Vicuna is created by fine-tuning a LLaMA base model using approximately 125K user-shared conversations gathered from ShareGPT.com with public APIs. To ensure data quality, we convert the HTML back to markdown and filter out some inappropriate or low-quality samples. Additionally, we divide lengthy conversations into smaller segments that fit the model's maximum context length. For detailed instructions to clean the ShareGPT data, check out [here](docs/commands/data_cleaning.md).

We will not release the ShareGPT dataset. If you would like to try the fine-tuning code, you can run it with some dummy conversations in [dummy_conversation.json](data/dummy_conversation.json). You can follow the same format and plug in your own data.

### Code and Hyperparameters

Our code is based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) with additional support for multi-turn conversations.
We use similar hyperparameters as the Stanford Alpaca.

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vicuna-13B | 128 | 2e-5 | 3 | 2048 | 0 |

### Fine-tuning Vicuna-7B with Local GPUs

- Install dependency

```bash
pip3 install -e ".[train]"
```

- You can use the following command to train Vicuna-7B with 4 x A100 (40GB). Update `--model_name_or_path` with the actual path to LLaMA weights and `--data_path` with the actual path to data.

```bash
torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path ~/model_weights/llama-7b  \
    --data_path data/dummy_conversation.json \
    --bf16 True \
    --output_dir output_vicuna \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```

Tips:

- If you are using V100 which is not supported by FlashAttention, you can use the [memory-efficient attention](https://arxiv.org/abs/2112.05682) implemented in [xFormers](https://github.com/facebookresearch/xformers). Install xformers and replace `fastchat/train/train_mem.py` above with [fastchat/train/train_xformers.py](fastchat/train/train_xformers.py).
- If you meet out-of-memory due to "FSDP Warning: When using FSDP, it is efficient and recommended... ", see solutions [here](https://github.com/huggingface/transformers/issues/24724#issuecomment-1645189539).
- If you meet out-of-memory during model saving, see solutions [here](https://github.com/pytorch/pytorch/issues/98823).

### Other models and LoRA support

More instructions to train other models (e.g., FastChat-T5) and use LoRA are in [docs/training.md](docs/training.md).

### Fine-tuning on Any Cloud with SkyPilot

[SkyPilot](https://github.com/skypilot-org/skypilot) is a framework built by UC Berkeley for easily and cost effectively running ML workloads on any cloud (AWS, GCP, Azure, Lambda, etc.).
Find SkyPilot documentation [here](https://github.com/skypilot-org/skypilot/tree/master/llm/vicuna) on using managed spot instances to train Vicuna and save on your cloud costs.

## Citation

The code (training, serving, and evaluation) in this repository is mostly developed for or derived from the paper below.
Please cite it if you find the repository helpful.

```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

We are also planning to add more of our research to this repository.
