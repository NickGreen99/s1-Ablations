import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# Load a small subset of the ServiceNow/repliqa dataset.
# Adjust the split or subset range as needed.
repliqa = load_dataset("ServiceNow/repliqa", split="repliqa_0[:1]")

model_id = "meta-llama/Llama-3.2-1B-Instruct"

# Initialize the tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
).to("cuda")

# Subclass the TextGenerationPipeline to customize preprocessing and postprocessing.
class CustomTextGenerationPipeline(TextGenerationPipeline):
    def preprocess(self, inputs, **kwargs):
        # Tokenize the input text with truncation to manage context length.
        return self.tokenizer(inputs, return_tensors="pt", truncation=True).to(self.device)
    
    def _forward(self, model_inputs, **kwargs):
        # Retrieve generation parameters.
        max_length = kwargs.get("max_length", 2048)
        temperature = kwargs.get("temperature", 0.7)
        # Generate text using the model.
        return self.model.generate(**model_inputs, max_length=max_length, temperature=temperature)
    
    def postprocess(self, model_outputs, **kwargs):
        # Decode the generated tokens back into text.
        generated_text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        return [{"generated_text": generated_text.strip()}]

# Instantiate the custom pipeline.
custom_generate = CustomTextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda")

# Process each example in the dataset.
for example in repliqa:
    # In this example, we assume the dataset contains an "instruction" and "input" field.
    # You might need to adjust the field names based on the dataset.
    instruction = example.get("question", "")
    context = example.get("document_extracted", "")
    
    # Combine the fields into a single prompt.
    prompt = (
        f"Instruction: {instruction}\n"
        f"Context: {context}\n"
        "Answer (Give exact quote from the context):"
    )
    
    # Generate a response using the custom pipeline.
    response = custom_generate(prompt, max_length=2048, temperature=0.7)
    print("Response:\n", response[0]["generated_text"])
    print("-" * 80)
