"""
The gradio demo server for chatting with a single model.
"""

import argparse
from collections import defaultdict
import datetime
import json
import os
import random
import time
import uuid

import gradio as gr
import requests

from fastchat.conversation import SeparatorStyle
from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INACTIVE_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.model.model_registry import model_info
from fastchat.serve.api_provider import (
    anthropic_api_stream_iter,
    openai_api_stream_iter,
    palm_api_stream_iter,
    init_palm_chat,
)
from fastchat.utils import (
    build_logger,
    violates_moderation,
    get_window_url_params_js,
    parse_gradio_auth_creds,
)
from fastchat.serve.openai_api_server import get_gen_params

import subprocess
import base64
import time
import hashlib

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.chat_vector_db.prompts import (QA_PROMPT)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "FastChat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

controller_url = None
enable_moderation = False

acknowledgment_md = """
<div class="image-container">
    <p> <strong>Acknowledgment: </strong> We thank <a href="https://www.kaggle.com/" target="_blank">Kaggle</a>, <a href="https://mbzuai.ac.ae/" target="_blank">MBZUAI</a>, <a href="https://www.anyscale.com/" target="_blank">AnyScale</a>, and <a href="https://huggingface.co/" target="_blank">HuggingFace</a> for their sponsorship. </p>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Kaggle_logo.png/400px-Kaggle_logo.png" alt="Image 1">
    <img src="https://mma.prnewswire.com/media/1227419/MBZUAI_Logo.jpg?p=facebookg" alt="Image 2">
    <img src="https://docs.anyscale.com/site-assets/logo.png" alt="Image 3">
    <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" alt="Image 4">
</div>
"""

ip_expiration_dict = defaultdict(lambda: 0)

# Information about custom OpenAI compatible API models.
# JSON file format:
# {
#     "vicuna-7b": {
#         "model_name": "vicuna-7b-v1.5",
#         "api_base": "http://8.8.8.55:5555/v1",
#         "api_key": "password"
#     },
# }
openai_compatible_models_info = {}

enable_attest = False


class State:
    def __init__(self, model_name):
        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name

        if model_name == "palm-2":
            # According to release note, "chat-bison@001" is PaLM 2 for chat.
            # https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023
            self.palm_chat = init_palm_chat("chat-bison@001")

    def to_gradio_chatbot(self):
        return self.conv.to_gradio_chatbot()

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )
        return base


class DocqaState:
    def __init__(self, model_name):
        self.model_name = model_name
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.all_splits = None
        self.docsearch = None


def set_global_vars(controller_url_, enable_moderation_):
    global controller_url, enable_moderation
    controller_url = controller_url_
    enable_moderation = enable_moderation_


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(
    controller_url, register_openai_compatible_models, add_chatgpt, add_claude, add_palm
):
    if controller_url:
        ret = requests.post(controller_url + "/refresh_all_workers")
        assert ret.status_code == 200
        ret = requests.post(controller_url + "/list_models")
        models = ret.json()["models"]
    else:
        models = []

    # Add API providers
    if register_openai_compatible_models:
        global openai_compatible_models_info
        openai_compatible_models_info = json.load(
            open(register_openai_compatible_models)
        )
        models += list(openai_compatible_models_info.keys())

    if add_chatgpt:
        models += ["gpt-3.5-turbo", "gpt-4"]
    if add_claude:
        models += ["claude-2", "claude-instant-1"]
    if add_palm:
        models += ["palm-2"]
    models = list(set(models))

    priority = {k: f"___{i:02d}" for i, k in enumerate(model_info)}
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


def load_demo_single_docqa(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown.update(
        choices=models, value=selected_model, visible=True
    )

    state = None
    return (
        state,
        dropdown_update,
        gr.File.update(visible=True),
        gr.Textbox.update(visible=True, interactive=False),
        gr.Textbox.update(visible=True, interactive=False),
        gr.Button.update(visible=True, interactive=False),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True)
    )


def load_demo_single_comp(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown.update(
        choices=models, value=selected_model, visible=True
    )

    return (
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo_single(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown.update(
        choices=models, value=selected_model, visible=True
    )

    state = None
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo_completion(url_params, request: gr.Request):
    global models

    ip = request.client.host
    logger.info(f"load_demo_completion. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    if args.model_list_mode == "reload":
        models = get_model_list(
            controller_url, args.add_chatgpt, args.add_claude, args.add_palm
        )

    return load_demo_single_comp(models, url_params)


def load_demo_docqa(url_params, request: gr.Request):
    global models

    ip = request.client.host
    logger.info(f"load_qa_demo_completion. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME
    if args.model_list_mode == "reload":
        models = get_model_list(
            controller_url, args.add_chatgpt, args.add_claude, args.add_palm
        )

    return load_demo_single_docqa(models, url_params)


def load_demo(url_params, request: gr.Request):
    global models

    ip = request.client.host
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    if args.model_list_mode == "reload":
        models = get_model_list(
            controller_url,
            args.register_openai_compatible_models,
            args.add_chatgpt,
            args.add_claude,
            args.add_palm,
        )

    return load_demo_single(models, url_params)


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.conv.update_last_message(None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (state, [], "") + (disable_btn,) * 5


def clear_completion_history(request: gr.Request):
    return ("", "") + (disable_btn,) * 2


def clear_docqa_history():
    state = None
    return (state, gr.File.update(value=None), "", "") + (disable_btn, ) * 2


def add_text(state, model_selector, text, request: gr.Request):
    ip = request.client.host
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5

    if ip_expiration_dict[ip] < time.time():
        logger.info(f"inactive. ip: {request.client.host}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), INACTIVE_MSG) + (no_change_btn,) * 5

    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), MODERATION_MSG) + (
                no_change_btn,
            ) * 5

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {request.client.host}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (
            no_change_btn,
        ) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


async def model_worker_completion_stream_iter(
    model_name,
    worker_addr,
    message,
    temperature,
    top_p,
    max_new_tokens,
):
    # Generate generate params
    gen_params = await get_gen_params(
        model_name,
        worker_addr,
        message,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        echo=False,
        stop=None,
    )

    # Print a log
    logger.info(f"==== request ====\n{gen_params}")

    # Send the request to the worker, and get response
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    print(response)
    # Handle the response, and decode the result
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def model_worker_stream_iter(
    conv,
    model_name,
    worker_addr,
    prompt,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
):
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    logger.info(f"==== request ====\n{gen_params}")

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def bot_response(state, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"bot_response. ip: {request.client.host}")
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    conv, model_name = state.conv, state.model_name
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_name == "claude-2" or model_name == "claude-instant-1":
        prompt = conv.get_prompt()
        stream_iter = anthropic_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_name == "palm-2":
        stream_iter = palm_api_stream_iter(
            state.palm_chat, conv.messages[-2][1], temperature, top_p, max_new_tokens
        )
    elif model_name in openai_compatible_models_info:
        model_info = openai_compatible_models_info[model_name]
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_info["model_name"],
            prompt,
            temperature,
            top_p,
            max_new_tokens,
            api_base=model_info["api_base"],
            api_key=model_info["api_key"],
        )
    else:
        # Query worker address
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (
                state,
                state.to_gradio_chatbot(),
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
            )
            return

        # Construct prompt.
        # We need to call it here, so it will not be affected by "▌".
        prompt = conv.get_prompt()

        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
        )

    conv.update_last_message("▌")
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        for i, data in enumerate(stream_iter):
            if data["error_code"] == 0:
                if i % 5 != 0:  # reduce gradio's overhead
                    continue
                output = data["text"].strip()
                conv.update_last_message(output + "▌")
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                )
                return
        output = data["text"].strip()
        if "vicuna" in model_name:
            output = post_process_code(output)
        conv.update_last_message(output)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def docqa_bot_response(state, msg, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"docqa_bot_response. ip: {request.client.host}")
    # start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    logger.info(f"Query: {msg}")
    llm = ChatOpenAI(model_name=state.model_name, temperature=temperature, max_tokens=max_new_tokens)
    doc_chain = load_qa_chain(
        llm, chain_type="stuff",prompt=QA_PROMPT
    )
    docs = state.docsearch.similarity_search(msg)
    output = doc_chain.run(input_documents=docs, question=msg)
    logger.info(f"Answer: {output}")
    # output = answer.strip()
    yield(state, output, no_change_btn)


block_css = """
#notice_markdown {
    font-size: 104%
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_markdown {
    font-size: 104%
}
#leaderboard_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_dataframe td {
    line-height: 0.1em;
}
footer {
    display:none !important
}
.image-container {
    display: flex;
    align-items: center;
    padding: 1px;
}
.image-container img {
    margin: 0 30px;
    height: 20px;
    max-height: 100%;
    width: auto;
    max-width: 20%;
}
"""


def get_model_description_md(models):
    model_description_md = """
| | | |
| ---- | ---- | ---- |
"""
    ct = 0
    visited = set()
    for i, name in enumerate(models):
        if name in model_info:
            minfo = model_info[name]
            if minfo.simple_name in visited:
                continue
            visited.add(minfo.simple_name)
            one_model_md = f"[{minfo.simple_name}]({minfo.link}): {minfo.description}"
        else:
            visited.add(name)
            one_model_md = (
                f"[{name}](): Add the description at fastchat/model/model_registry.py"
            )

        if ct % 3 == 0:
            model_description_md += "|"
        model_description_md += f" {one_model_md} |"
        if ct % 3 == 2:
            model_description_md += "\n"
        ct += 1
    return model_description_md


def attest(user_data):
    from bigdl.ppml.attestation import attestation_service, quote_generator

    cur_timestamp = str(int(time.time()))
    report_data_base = user_data + cur_timestamp
    sha256 = hashlib.sha256()
    sha256.update(report_data_base.encode())
    user_report_data = sha256.hexdigest()
    try:
        quote_b = quote_generator.generate_tdx_quote(user_report_data)
        base64_data = base64.b64encode(quote_b).decode("utf-8")
    except Exception as e:
        base64_data = "Gradio web server generate quote failed: %s" % (e)
    header_off = 0
    report_body_off = 48
    report_data_len = 64
    quote_list = []

    ret = requests.post(controller_url + "/attest", json={"userdata": user_data})
    assert ret.status_code == 200
    quote_ret = ret.json()["quote"]
    quote_list.append(["controller", "Unverified", quote_ret])

    ret = requests.post(controller_url + "/refresh_all_workers")
    assert ret.status_code == 200

    ret = requests.post(
        controller_url + "/attest_workers", json={"userdata": user_data}
    )
    assert ret.status_code == 200
    workers_quote_ret = ret.json()["quote_list"]
    for worker_name, worker_quote in workers_quote_ret.items():
        quote_list.append(["worker-%s" % worker_name, "Unverified", worker_quote])

    return base64_data, quote_list, cur_timestamp


def verify(as_url, as_app_id, as_api_key, quote_list):
    from bigdl.ppml.attestation import attestation_service, quote_generator

    print(quote_list)
    for index, quote_row in quote_list.iterrows():
        print(quote_row)
        attestation_result = attestation_service.bigdl_attestation_service(
            as_url, as_app_id, as_api_key, base64.b64decode(quote_row["quote"]), ""
        )
        if attestation_result >= 0:
            quote_row["status"] = "Attestation Successed"
        else:
            quote_row["status"] = "Attestation Failed"
    return quote_list


def ingest(state, file):
    if state is None:
        state = DocqaState("gpt-3.5-turbo")
    # Process File
    logger.info(f"File Name: {file.name}")
    if file.name.endswith(".txt"):
        loader = TextLoader(file.name)
    elif file.name.endswith(".pdf"):
        loader = PyPDFLoader(file.name)
    state.all_splits = loader.load_and_split(state.text_splitter)
    state.docsearch = Chroma.from_documents(documents=state.all_splits, embedding=state.embedding)
    return (state, gr.Textbox.update(interactive=True)) + (enable_btn, ) * 2


# Return button states
async def bot_completion(
    msg, model_name, temperature, top_p, max_new_tokens, request: gr.Request
):
    logger.info(f"bot_completion. ip: {request.client.host}")
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"Completion req model_name: {model_name}, worker_addr: {worker_addr}")

    # Handle no available worker
    if worker_addr == "":
        yield (
            SERVER_ERROR_MSG,
            enable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Now let's use the worker for completions
    # completion_stream_iter = await model_worker_completion_stream_iter(
    #     model_name, worker_addr, msg, temperature, top_p, max_new_tokens
    # )

    try:
        async for data in model_worker_completion_stream_iter(
            model_name, worker_addr, msg, temperature, top_p, max_new_tokens
        ):
            if data["error_code"] == 0:
                output = data["text"].strip()
                yield (output, disable_btn, disable_btn, disable_btn)
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                yield (output, enable_btn, enable_btn, enable_btn)
                return
    except requests.exceptions.RequestException as e:
        output = (
            f"{SERVER_ERROR_MSG}\n\n(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )

        yield (output, enable_btn, enable_btn, enable_btn)
        return
    except Exception as e:
        output = f"{SERVER_ERROR_MSG}\n\n(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"

        yield (output, enable_btn, enable_btn, enable_btn)
        return

    yield (output, enable_btn, enable_btn, enable_btn)

    return


def build_completion_mode_ui(models, add_promotion_links=False):
    promotion = (
        """
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)
- | [GitHub](https://github.com/lm-sys/FastChat) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |
"""
        if add_promotion_links
        else ""
    )

    # TODO: change this terms of use
    notice_markdown = f"""
# 🏔️ Completion with Open Large Language Models and bigdl-llm support
{promotion}

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.**

### Choose a model to chat with
"""

    model_description_md = get_model_description_md(models)
    gr.Markdown(notice_markdown + model_description_md, elem_id="notice_markdown")

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
            container=False,
        )

    # Response box for completion
    response_textbox = gr.Textbox(
        show_label=True,
        label="Response",
        height=400,
        visible=False,
    )

    # Let user enter prompt and place the send button
    with gr.Row():
        with gr.Column(scale=15):
            input_textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
                container=False,
            )
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)
    with gr.Row() as button_row:
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    btn_list = [regenerate_btn, clear_btn]
    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    model_selector.change(
        clear_completion_history, None, [input_textbox, response_textbox] + btn_list
    )
    clear_btn.click(
        clear_completion_history, None, [input_textbox, response_textbox] + btn_list
    )
    input_textbox.submit(
        bot_completion,
        [input_textbox, model_selector, temperature, top_p, max_output_tokens],
        [response_textbox] + btn_list + [send_btn],
    )

    regenerate_btn.click(
        bot_completion,
        [input_textbox, model_selector, temperature, top_p, max_output_tokens],
        [response_textbox] + btn_list + [send_btn],
    )

    send_btn.click(
        bot_completion,
        [input_textbox, model_selector, temperature, top_p, max_output_tokens],
        [response_textbox] + btn_list + [send_btn],
    )

    return (
        model_selector,
        response_textbox,
        input_textbox,
        send_btn,
        button_row,
        parameter_row,
    )


def build_single_model_ui(models, add_promotion_links=False):
    promotion = (
        """
- Introducing Llama 2: The Next Generation Open Source Large Language Model. [[Website]](https://ai.meta.com/llama/)
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)
- | [GitHub](https://github.com/lm-sys/FastChat) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |
"""
        if add_promotion_links
        else ""
    )

    notice_markdown = f"""
# 🏔️ Chat with Open Large Language Models
{promotion}

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.**

### Choose a model to chat with
"""

    state = gr.State()
    model_description_md = get_model_description_md(models)
    gr.Markdown(notice_markdown + model_description_md, elem_id="notice_markdown")

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
            container=False,
        )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        label="Scroll down and start chatting",
        visible=False,
        height=550,
    )
    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter your prompt here and press ENTER",
                visible=False,
                container=False,
            )
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False, variant="primary")

    with gr.Row(visible=False) as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    if add_promotion_links:
        gr.Markdown(acknowledgment_md)

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    textbox.submit(
        add_text, [state, model_selector, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text, [state, model_selector, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    return state, model_selector, chatbot, textbox, send_btn, button_row, parameter_row


def build_attestation_ui(models):
    with gr.Accordion(
        label="Remote Attestation", elem_id="attestation_panel", open=True
    ) as remote_attestation_app:
        with gr.Row().style(equal_height=True):
            # with gr.Column(scale=0.4, min_width=0):
            #     tee_status = gr.Text(label="TEE Status", value="TDX enabled")
            with gr.Column(scale=0.1, min_width=0):
                user_data = gr.Textbox("", show_label=True, label="User Data").style(
                    container=True
                )
            with gr.Column(scale=0.1, min_width=0):
                quote_timestamp = gr.Textbox(
                    "", show_label=True, label="Timestamp"
                ).style(container=True)
            with gr.Column(scale=0.7, min_width=0):
                quote = gr.Textbox(
                    label="Quote (BASE64 formatted)", placeholder="Unknown", max_lines=1
                ).style(container=True, show_copy_button=True)
            with gr.Column(scale=0.1, min_width=0):
                attest_btn = gr.Button("Generate Quote").style(container=True)

        with gr.Row():
            gr.Markdown(
                "<small> <em> The Quote Generation will use SHA-256 Hash of User Data concatted with Timestamp as the report data. </em> </small>"
            )

        with gr.Accordion(label="Quote Details:", open=True):
            quote_df = gr.Dataframe(
                headers=["role", "status", "quote"],
                datatype=["str", "str", "str"],
                value=[["", "", ""]],
                col_count=(3, "fixed"),
                interactive=False,
            )

        with gr.Row():
            as_url = gr.Textbox(
                "", show_label=True, label="Attestation Service URL"
            ).style(container=True)
            as_app_id = gr.Textbox(
                "", show_label=True, label="Attestation App ID"
            ).style(container=True)
            as_api_key = gr.Textbox(
                "", show_label=True, type="password", label="Attestation Api Key"
            ).style(container=True)
            with gr.Column(scale=0.2, min_width=0):
                verify_btn = gr.Button("Attest with AS").style(container=True)
            # with gr.Column(scale=0.1, min_width=0):
            # verify_btn = gr.Button("Verify")

    attest_btn.click(
        attest, [user_data], [quote, quote_df, quote_timestamp], queue=False
    )
    verify_btn.click(
        verify, [as_url, as_app_id, as_api_key, quote_df], [quote_df], queue=False
    )


def build_docqa_model_ui(models, add_promotion_links=False):
    promotion = (
        """
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)
- | [GitHub](https://github.com/lm-sys/FastChat) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |
"""
        if add_promotion_links
        else ""
    )

    # TODO: change this terms of use
    notice_markdown = f"""
# 📁 Interpreting Documents Using Large Language Models
{promotion}
### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.**
### Choose a model to chat with
"""

    state = gr.State()
    model_description_md = get_model_description_md(models)
    gr.Markdown(notice_markdown + model_description_md, elem_id="notice_markdown")

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
            container=False,
        )

    #Upload file component
    file_uploader = gr.File(
        file_types=['.pdf', '.txt'],
        visible=False,
        label="Please upload your file",
    )

    response_textbox = gr.Textbox(
        show_label=True,
        label="Response",
        visible=False,
    )

    # Let user enter prompt and place the send button
    with gr.Row():
        with gr.Column(scale=20):
            qa_input_textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
                container=False,
            )
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)
    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    btn_list = [clear_btn]
    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    #Register listeners
    model_selector.change(
        clear_docqa_history, None, [state, file_uploader, response_textbox, qa_input_textbox] + btn_list + [send_btn]
    )
    clear_btn.click(
        clear_docqa_history, None, [state, file_uploader, response_textbox, qa_input_textbox] + btn_list + [send_btn]
    )
    file_uploader.clear(
        clear_docqa_history, None, [state, file_uploader, response_textbox, qa_input_textbox] + btn_list + [send_btn]
    )

    file_uploader.upload(
        ingest,
        [state, file_uploader],
        [state, qa_input_textbox] + btn_list + [send_btn],
    )

    qa_input_textbox.submit(
        docqa_bot_response,
        [state, qa_input_textbox, temperature, top_p, max_output_tokens],
        [state, response_textbox] + btn_list,
    )
    send_btn.click(
        docqa_bot_response,
        [state, qa_input_textbox, temperature, top_p, max_output_tokens],
        [state, response_textbox] + btn_list,
    )

    return (
        state,
        model_selector,
        file_uploader,
        response_textbox,
        qa_input_textbox,
        send_btn,
        button_row,
        parameter_row,
    )


def build_demo(models):
    with gr.Blocks() as demo:
        with gr.Tab("Chat"):
            with gr.Blocks(
                title="Chat with Open Large Language Models",
                theme=gr.themes.Base(),
                css=block_css,
            ):
                url_params = gr.JSON(visible=False)

                (
                    state,
                    model_selector,
                    chatbot,
                    textbox,
                    send_btn,
                    button_row,
                    parameter_row,
                ) = build_single_model_ui(models)

                if args.model_list_mode not in ["once", "reload"]:
                    raise ValueError(f"Unknown model list mode: {args.model_list_mode}")
                demo.load(
                    load_demo,
                    [url_params],
                    [
                        state,
                        model_selector,
                        chatbot,
                        textbox,
                        send_btn,
                        button_row,
                        parameter_row,
                    ],
                    _js=get_window_url_params_js,
                )

        with gr.Tab("Completion"):
            with gr.Blocks(
                title="Completion using Large language Models",
                theme=gr.themes.Base(),
                css=block_css,
            ):
                url_params = gr.JSON(visible=False)

                (
                    model_selector,
                    response_textbox,
                    input_textbox,
                    send_btn,
                    button_row,
                    parameter_row,
                ) = build_completion_mode_ui(models)

                if args.model_list_mode not in ["once", "reload"]:
                    raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

                demo.load(
                    load_demo_completion,
                    [url_params],
                    [
                        model_selector,
                        response_textbox,
                        input_textbox,
                        send_btn,
                        button_row,
                        parameter_row,
                    ],
                    _js=get_window_url_params_js,
                )

        if enable_attest:
            with gr.Tab("Attestation"):
                with gr.Blocks(
                    title="Remote Attestation",
                    theme=gr.themes.Base(),
                    css=block_css,
                ):
                    url_params = gr.JSON(visible=False)
                    build_attestation_ui(models)
        
        with gr.Tab("Document QA"):
            with gr.Blocks(
                title="Interpreting Documents Using Large Language Models",
                theme = gr.themes.Base(),
                css=block_css,
            ):
                url_params = gr.JSON(visible=False)
                (
                    state,
                    model_selector,
                    file_uploader,
                    response_textbox,
                    qa_input_textbox,
                    send_btn,
                    button_row,
                    parameter_row,
                ) = build_docqa_model_ui(models)

                if args.model_list_mode not in ["once", "reload"]:
                    raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

                demo.load(
                    load_demo_docqa,
                    [url_params],
                    [
                        state,
                        model_selector,
                        file_uploader,
                        response_textbox,
                        qa_input_textbox,
                        send_btn,
                        button_row,
                        parameter_row,
                    ],
                    _js=get_window_url_params_js,
                )

    return demo


if __name__ == "__main__":
    # use faux openai server
    # Get the value of the OPENAI_API_BASE environment variable, or None if it's not set
    api_base = os.environ.get('OPENAI_API_BASE')
    # Check if the environment variable is already set
    if api_base is None:
        # If it's not set, set the default address
        os.environ['OPENAI_API_BASE'] = 'http://localhost:8000/v1'
        api_base = 'http://localhost:8000/v1'

    os.environ['OPENAI_API_KEY'] = 'EMPTY'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link.",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue.",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time.",
    )
    parser.add_argument(
        "--moderate", action="store_true", help="Enable content moderation"
    )
    parser.add_argument(
        "--add-chatgpt",
        action="store_true",
        help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        help="Add Anthropic's Claude models (claude-2, claude-instant-1)",
    )
    parser.add_argument(
        "--add-palm",
        action="store_true",
        help="Add Google's PaLM model (PaLM 2 for Chat: chat-bison@001)",
    )
    parser.add_argument(
        "--register-openai-compatible-models",
        type=str,
        help="Register custom OpenAI API compatible models by loading them from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    )
    parser.add_argument(
        "--attest", action="store_true", help="whether enable attesation"
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate)
    models = get_model_list(
        args.controller_url,
        args.register_openai_compatible_models,
        args.add_chatgpt,
        args.add_claude,
        args.add_palm,
    )
    enable_attest = args.attest

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
    )
