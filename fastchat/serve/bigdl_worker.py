"""
A model worker that executes the model based on BigDL-LLM.

TODO: add documentation
"""

import argparse
import asyncio
import atexit
import json
from typing import List
import uuid
from threading import Thread
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.serve.base_model_worker import (
    create_background_tasks,
)
from fastchat.utils import get_context_length, is_partial_stop

from bigdl.llm.transformers.loader import load_model
from transformers import TextIteratorStreamer

app = FastAPI()

# TODO: decide if we need stream_interval

class BigDLLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str = None,
        load_in_low_bit: str = 'sym_int4',
        device: str = 'cpu',
        no_register: bool = False,
        # TODO: multimodel
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        self.load_in_low_bit = load_in_low_bit
        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: BigDLLLM worker..."
        )

        logger.info(
            f"Using low bit format: {self.load_in_low_bit}, device: {device}"
        )

        self.model, self.tokenizer = load_model(model_path, device, self.load_in_low_bit)
        self.context_len = get_context_length(self.model.config)
        if not no_register:
            self.init_heart_beat()
        
    # TODO: uncomment
    #async def generate_stream(self, params):
    def generate_stream(self, params):
        self.call_ct += 1
        # context length is self.context_length
        prompt = params["prompt"]
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 256))
        # TODO: logprobs probably not supported
        logprobs = params.get("logprobs", None)
        echo = bool(params.get("echo", True))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(self.tokenizer.eos_token_id)

        # Add stop string to stop set
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        # We no longer need to handle stop_token_ids
        # But only stop
        for tid in stop_token_ids:
            if tid is not None:
                s = self.tokenizer.decode(tid)
                if s != "":
                    stop.add(s)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        if self.model.config.is_encoder_decoder:
            max_src_len = self.context_len
        else:
            max_src_len = self.context_len - max_new_tokens

        input_ids = input_ids[-max_src_len:]

        input_echo_len = len(input_ids)


        ####################Preparation is done ######################################


        # 2. Generate a generator, then for it to generate the result


        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        # Possible generation config:
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        generated_kwargs = dict(
            max_new_tokens = max_new_tokens,
            streamer = streamer,
            temperature = temperature,
            repetition_penalty = repetition_penalty,
            top_p = top_p,
            top_k = top_k,
        )

        def model_generate():
            self.model.generate(input_ids, **generated_kwargs)

        t1 = Thread(target=model_generate)
        t1.start()

        # partial_text = ""
        # for next_text in streamer:
        #     partial_text += next_text
        #     # print(next_text)

        """
        Logic for the entire procedure:
        1. Get the output token
        2. (Optional) Logprobs
        3. Check if token in stop_token_ids
            If yes, set stopped to True
            Otherwise set stopped to False
        4. Yield the output tokens
            Check stream_interval or max_tokens or stopped
            (Not needed) decode the token
            (Optional) logprobs processed
            (Optional) judge_sent_end
            Check stop_str -> set stopped to true
            Check partially_stopped -> if partially stopped, continue
                else yield the token
            Check stopped, break

            for loop ends, finish reason="length"
            otherwise, finish_reason="stop"
            yield last one
        """
        stopped = False
        finish_reason = None
        if echo:
            partial_output = prompt
            rfind_start = len(prompt)
        else:
            partial_output = ""
            rfind_start = 0

        for i in range(max_new_tokens):
            # Get a token from the streamer
            try:
                output_token = next(streamer)
            except StopIteration:
                    # TODO: handle this StopIteration
                    print("Reached StopIteration")
                    break

                # Check if this new token is in stop
            partial_output += output_token
            for each_stop in stop:
                pos = partial_output.rfind(each_stop, rfind_start)
                if pos != -1:
                    partial_output = partial_output[:pos]
                    stopped = True
                    break
                else:
                    partially_stopped = is_partial_stop(partial_output, each_stop)
                    if partially_stopped:
                        break
            if not partially_stopped:
                json_output = {
                    "text": partial_output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }
                ret = {
                    "text": json_output["text"],
                    "error_code": 0,
                }
                ret["usage"] = json_output["usage"]
                ret["finish_reason"] = json_output["finish_reason"]
                yield json.dumps(ret).encode() + b"\0"

            if stopped:
                break
        else:
            finish_reason = "length"

        if stopped:
            finish_reason = "stop"

        json_output = {
            "text": partial_output,
            "error_code": 0,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }
        # ret = {
        #     "text": json_output["text"],
        #     "error_code": 0,
        # }
        # ret["usage"] = json_output["usage"]
        # ret["finish_reason"] = json_output["finish_reason"]
        yield json.dumps(json_output).encode() + b"\0"



        # Get the result from the streamer
        # TODO: uncomment
        # for next_text in streamer:
        #     print(next_text)
        # for next_text in streamer:
        #     if echo:
        #         text_outputs = [
        #             prompt + next_text
        #         ]
        #     else:
        #         text_outputs = [next_text]
        #     text_outputs = " ".join(text_outputs)

        #     partial_stop = any(is_partial_stop(text_outputs, i) for i in stop)

        #     if partial_stop:
        #         continue

        #     # aborted = False
        #     ret = {
        #         "text": text_outputs,
        #         "error_code": 0,
        #         "usage": {
        #             "prompt_tokens": prompt_tokens,
        #         }
        #     }
        # pass

def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()

def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks



########################## We would need to implement the following APIs############################
@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)

# @app.post("/worker_generate")
# async def api_generate(request: Request):
#     params = await request.json()
#     await acquire_worker_semaphore()
#     request_id = random_uuid()
#     params["request_id"] = request_id
#     params["request"] = request
#     output = await worker.generate(params)
#     release_worker_semaphore()
#     await engine.abort(request_id)
#     return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}





# TODO: modify
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    # TODO: add options
    parser.add_argument(
        "--low-bit", type=str, default='sym_int4', help="Low bit format."
    )
    # TODO: add options for possible quantization type
    parser.add_argument(
        "--device", type=str, default='cpu', help="Device for executing model, cpu/xpu"
    )
    # TODO: we might want to enable this instead of setting it to default
    # parser.add_argument(
    #     "--trust_remote_code",
    #     action="store_false",
    #     default=True,
    #     help="Trust remote code (e.g., from HuggingFace) when"
    #     "downloading the model and tokenizer.",
    # )

    args = parser.parse_args()
    worker = BigDLLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.conv_template,
        args.low_bit,
        args.device,
        args.no_register,
    )

    # params = {
    #     "prompt": "What is AI?",
    # }
    # generator = worker.generate_stream(params)
    # for jsons in generator:
    #     print(jsons)

    # TODO: uncomment
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
