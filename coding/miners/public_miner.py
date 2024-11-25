import os
import json
import time
import random
import bittensor as bt
from functools import partial
from starlette.types import Send
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from typing import List, Any, Dict, Awaitable
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence

from coding.protocol import StreamCodeSynapse
from coding.helpers import extract_python_code

import difflib
import re


def clean_diff_content(diff_content):
    # Remove diff-specific lines
    cleaned_lines = []
    for line in diff_content.splitlines():
        if not line.startswith(("@@", "---", "+++", "diff --git", "index")):
            cleaned_lines.append(line[1:] if line.startswith(("+", "-")) else line)
    return "\n".join(cleaned_lines).strip()


def rename_diff_files(diff_dict, files):
    renamed_dict = {}

    for file in files:
        for diff_filename, diff_content in diff_dict.items():
            cleaned_diff = clean_diff_content(diff_content)
            # Calculate similarity ratio between cleaned diff and file content
            similarity = difflib.SequenceMatcher(
                None, cleaned_diff, file.content
            ).ratio()
            if similarity > 0.9:  # Adjust this threshold as needed
                renamed_dict[file.path] = diff_content
                break
        else:
            # If no match is found, retain the original filename
            renamed_dict[diff_filename] = diff_content

    return renamed_dict


def parse_diff(diff_string):
    lines = diff_string.splitlines()
    file_diffs = {}
    current_file = None
    diff_content = []
    is_diff_block = False

    for line in lines:
        if "diff --git" in line:
            if current_file and diff_content:
                file_diffs[current_file] = "\n".join(diff_content)
            current_file = line.split()[-1]
            diff_content = []
            is_diff_block = False
        elif line.startswith("---") or line.startswith("+++"):
            # Ignore these lines, as they indicate the old/new file path
            continue
        elif line.startswith("@@"):
            is_diff_block = True
            continue
        elif is_diff_block:
            diff_content.append(line)

    if current_file and diff_content:
        file_diffs[current_file] = "\n".join(diff_content)

    return file_diffs


def render_mistral_template(messages, bos_token="[INST]", eos_token="[/INST]"):
    if messages[0].role == "system":
        system_message = messages[0].content.strip() + "\n\n"
        messages = messages[1:]
    else:
        system_message = ""

    result = bos_token + system_message

    for index, message in enumerate(messages):
        if (message.role == "user") != (index % 2 == 0):
            raise Exception(
                "Conversation roles must alternate user/assistant/user/assistant/..."
            )

        if message.role == "user":
            result += bos_token + " " + message.content.strip() + " " + eos_token
        elif message.role == "assistant":
            result += bos_token + " " + message.content.strip() + eos_token

    return result


def miner_init(self):
    """
    Initializes the miner. This function is called once when the miner is created.
    """
    _ = load_dotenv(find_dotenv())
    api_key = os.environ.get("OPENAI_API_KEY", "NONE")

    def model_factory(
        api_base="http://localhost:8000/v1",
        model_name=self.config.neuron.model_id,
        max_tokens=2048,
        temperature=0.7,
        top_p=1.0,
        chat=False,
    ):
        if chat:
            return ChatOpenAI(
                openai_api_base=api_base,
                openai_api_key=api_key,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                streaming=True,
            )
        return OpenAI(
            openai_api_base=api_base,
            openai_api_key=api_key,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            streaming=True,
        )

    self.model_factory = model_factory
    # Set openai key and other args
    self.model = OpenAI(
        openai_api_base="http://localhost:8000/v1",
        openai_api_key=api_key,
        model_name=self.config.neuron.model_id,
        max_tokens=2048,
        temperature=0.7,
        streaming=True,
    )


def miner_process(self, synapse: StreamCodeSynapse) -> Awaitable:
    """
    The miner process function is called every time the miner receives a request. This function should contain the main logic of the miner.
    """
    self.model = OpenAI(
        openai_api_base=f"http://localhost:8000/v1",
        openai_api_key="EMPTY",
        model_name=self.config.neuron.model_id,
        max_tokens=2048,
        temperature=0.7,
        streaming=True,
    )

    async def _forward(
        self,
        query: str,
        files: List[Any],
        extra_info: Dict[str, Any],
        init_time: float,
        timeout_threshold: float,
        chain: RunnableSequence,
        chain_formatter: Dict[str, str],
        send: Send,
    ):
        buffer = []
        temp_completion = ""  # for wandb logging
        timeout_reached = False

        try:
            # Langchain built in streaming. 'astream' also available for async
            for token in chain.stream(chain_formatter):
                if not isinstance(token, str):
                    token = token.content
                buffer.append(token)

                if time.time() - init_time > timeout_threshold:
                    bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                    timeout_reached = True
                    break

                if (
                    not "broken_file" in extra_info.keys()
                    and len(buffer) == self.config.neuron.streaming_batch_size
                ):
                    joined_buffer = "".join(buffer)
                    temp_completion += joined_buffer
                    bt.logging.debug(f"Streamed tokens: {repr(joined_buffer)}")

                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer,
                            "more_body": True,
                        }
                    )
                    buffer = []

            if (
                buffer and not timeout_reached
            ):  # Don't send the last buffer of data if timeout.
                body = "".join(buffer)
                if "broken_file" in extra_info.keys():
                    code = extract_python_code(body)
                    body = json.dumps(
                        {"path": extra_info["broken_file"]["path"], "content": code}
                    )
                await send(
                    {
                        "type": "http.response.body",
                        "body": body,
                        "more_body": False,
                    }
                )

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            if self.config.neuron.stop_on_forward_exception:
                self.should_exit = True

    if synapse.messages:
        query = synapse.messages[-1].content

    extra_info = {}
    model = self.model
    stop = None
    chain = None
    chain_formatter = None
    print("prompt", synapse.query)
    print("files", synapse.files)
    bt.logging.debug(f"📧 Query received, forwarding synapse: {synapse}")
    if "<|fim_hole|>" in synapse.query:
        synapse.query = synapse.query.replace("<|fim_hole|>", "<fim_suffix>")
        prompt = PromptTemplate.from_template("<fim_prefix>{query}<fim_middle>")
        # model = self.model_factory(max_tokens=1024, temperature=0.01, top_p=1.0)
        stop = [
            "<fim_prefix>",
            "<fim_suffix>",
            "<fim_middle>",
            "//",
            "<｜end▁of▁sentence｜>",
            "\n\n",
            "\r\n\r\n",
            "/src/",
            "#- coding: utf-8",
            "```",
            "\ndef",
            "\nclass",
            '\n"""#',
        ]
    elif synapse.files and "error" in synapse.files[0]:
        model = self.model_factory(
            api_base="http://localhost:8001/v1",
            model_name="thesven/Mistral-7B-Instruct-v0.3-GPTQ",
            max_tokens=2048,
            temperature=0.5,
            top_p=1.0,
        )
        string = ""
        # for att in synapse.files:

        # string += f"#{att['title']}\n{att['content']}\n"
        error = synapse.files[0]["error"].replace("}", "}}").replace("{", "{{")
        string += f"### Instruction:\nGiven the code: {{query}}, and the error: {error} rewrite the following file content to fix it. Where `# BROKEN` is, is where the fix is needed. Replace all the `# BROKEN`'s with the fixed file:"
        broken_file = [
            att for att in synapse.files[0]["files"] if "# BROKEN" in att.content
        ][0]
        extra_info["broken_file"] = broken_file
        broken_file.content = broken_file.content.replace("}", "}}").replace("{", "{{")
        string += f"\n ```python\n{broken_file['content']}\n```\n### Response: "
        prompt = PromptTemplate.from_template(string)
    elif (
        synapse.messages and synapse.files
    ):  # TODO messages is never used by the tasking, but it needs to be
        model = self.model_factory(
            api_base="http://localhost:8001/v1",
            model_name="thesven/Mistral-7B-Instruct-v0.3-GPTQ",
            max_tokens=2048,
            temperature=0.5,
            top_p=1.0,
        )
        string = render_mistral_template(synapse.messages)
        string += "[INST]"
        for file in synapse.files:
            file.content = file.content.replace("}", "}}").replace("{", "{{")
            string += f"#{file.path}\n{file.content}\n"
        string += "[/INST]{query}"
        query = ""
        prompt = PromptTemplate.from_template(string)
    elif (
        synapse.messages
    ):  # TODO messages is never used by the tasking, but it needs to be
        chain = self.model_factory(
            api_base="http://localhost:8001/v1",
            model_name="thesven/Mistral-7B-Instruct-v0.3-GPTQ",
            max_tokens=2048,
            temperature=0.5,
            top_p=1.0,
            chat=True,
        )
        synapse.messages[0].role = "user"
        chain_formatter = [msg.dict() for msg in synapse.messages]
        # string = render_mistral_template(synapse.messages) + "{query}"
        # query = ""
        # prompt = PromptTemplate.from_template(string)
    elif "The following issue is:\n\n" in synapse.query:
        # this is a SWE-Bench style task
        prompt = synapse.query + "\n"
        for file in synapse.files:
            prompt += f"#Filename: {file.path}\n{file.content}\n"
        prompt += (
            "Respond only with the patch, only modify the files you have been provided."
        )
        model = self.model_factory(
            api_base="http://localhost:8001/v1",
            model_name="thesven/Mistral-7B-Instruct-v0.3-GPTQ",
            max_tokens=2048,
            temperature=0.6,
            top_p=1.0,
            chat=True,
        )
        model_res = (
            model.invoke([{"role": "user", "content": prompt[0:15000]}])
            .content.replace("<patch>", "")
            .replace("</patch>", "")
            .replace("b/", "")
            .replace("a/", "")
        )
        if "```" in model_res:
            model_res = model_res.split("```")[1]
        print("prompt ---=--=-=--=-", prompt)
        print("🍊🍊🍊🍊 model response", model_res)
        model_res = json.dumps(parse_diff(model_res))
        print("🍊🍊🍊🍊🍊🍊🍊🍊 🍊🍊🍊🍊 🍊🍊🍊🍊  model response", model_res)

        async def _return(string, send: Send):
            await send(
                {
                    "type": "http.response.body",
                    "body": string,
                    "more_body": False,
                }
            )

        return synapse.create_streaming_response(partial(_return, model_res))
    elif synapse.files:
        string = ""
        for file in synapse.files:
            if "path" not in file:
                file.path = ""
            string += f"#{file.path}\n{file.content}\n"
        string += "{query}"
        prompt = PromptTemplate.from_template(string)
    else:
        # prompt = PromptTemplate.from_template(
        #     "{query}"
        # )
        prompt = synapse.query
        model = ChatOpenAI(
            openai_api_base=f"http://localhost:8000/v1",
            openai_api_key="EMPTY",
            model_name=self.config.neuron.model_id,
            max_tokens=2048,
            temperature=0.7,
            streaming=True,
        )

        async def _return(prompt, send: Send):
            for chunk in model.stream([{"role": "user", "content": prompt}]):
                print("string chunk", chunk.content)
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk.content,
                        "more_body": True,
                    }
                )
            await send(
                    {
                        "type": "http.response.body",
                        "body": chunk.content,
                        "more_body": False,
                    }
                )

        return synapse.create_streaming_response(partial(_return, prompt))
    if stop:
        model = model.bind(stop=stop)
    if not chain:
        chain = prompt | model

    query = synapse.query
    if not chain_formatter:
        chain_formatter = {"query": query}

    init_time = time.time()
    timeout_threshold = synapse.timeout

    token_streamer = partial(
        _forward,
        self,
        query,
        synapse.files,
        extra_info,
        init_time,
        timeout_threshold,
        chain,
        chain_formatter,
    )
    return synapse.create_streaming_response(token_streamer)
