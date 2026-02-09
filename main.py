from smolagents.models import LiteLLMModel
from transformers import AutoTokenizer
import os

os.environ["OPENAI_API_KEY"] = "fake"

# Use LiteLLMModel Instead of InferenceClientModel
# To use LiteLLMModel module in smolagents, you may run pip command to install the module.
    
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

prompt = tokenizer.apply_chat_template(chat,tokenize=False, add_generation_prompt=True,)
print(prompt)
# Above is just en example of Chat Format handling w/o using Jinja2 format by just exploiting HF tokenizers
# There is nothing to apply for gpt-oos-20b so the example is left just for a reason you need to use some particular instruct model
# like Mistral-7b-Instuct 
# You can simply transform the print(prompt) output to Jinja2 Template to use it somewhere outside like LM Studio for example
# Altirnatively you can use https://huggingface.co/spaces/Jofthomas/Chat_template_viewer to visually check and simply catch the formatted 
# message for ANY model by just referencing its hugging face modelname ex:'meta-llama/Meta-Llama-3-8B-Instruct'.

model = LiteLLMModel(
    model_id="openai/gpt-oss-20b",  # Or try other Ollama-supported models
    api_base="http://192.168.8.19:1234/v1", 
    api_key="fake",
    num_ctx=8192,
)

# response = model(
#     messages=[
#         {"role": "user", "content": "Explain what an AI agent is in one sentence."}
#     ]
# )

# print(response)
