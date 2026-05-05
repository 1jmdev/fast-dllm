from fastdllm.generation import generate_block_diffusion
from fastdllm.utils import load_model_and_tokenizer, set_seed

set_seed(42)
model, tokenizer = load_model_and_tokenizer("outputs/smollm2-135m-fastdllm/final", dtype="auto")
result = generate_block_diffusion(
    model,
    tokenizer,
    "Explain block diffusion language models in one paragraph.",
    max_new_tokens=96,
    block_size=32,
    sub_block_size=8,
    threshold=0.9,
)
print(result.text)
print(result.stats)
