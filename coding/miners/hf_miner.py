from typing import Awaitable
from transformers import AutoTokenizer, AutoModelForCausalLM

from coding.protocol import CodeSynapse
from langchain_huggingface.llms import HuggingFacePipeline


def miner_init(self):
    """
    Initializes the miner. This function is called once when the miner is created.
    """

    self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True).cuda()


    
def miner_process(self, synapse: CodeSynapse) -> Awaitable:
    """
    The miner process function is called every time the miner receives a request. This function should contain the main logic of the miner.
    """
    prompt = f'<｜fim▁begin｜>{synapse.query}<｜fim▁end｜>'
    
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    synapse.completion = self.model.generate(**inputs, max_length=4096)[0]
    self.bt.logging.info(f"HF LLM Response:{synapse.completion}")

    return synapse
