import torch
import chainlit
from transformers import AutoTokenizer

from llm_compressor.models.llama import CompressLlamaForCausalLM


############### Model Definition ###############
model_path = "d:\\models\\llama-3.2-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = CompressLlamaForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

############### Interact with Instruct-Model ###############
@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """

    response = model.generate_text(message.content, tokenizer)

    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface
    ).send()