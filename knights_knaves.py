import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# Load a small subset of the Knights and Knaves dataset
knight_knaves = load_dataset('K-and-K/knights-and-knaves', name="test", split="2ppl[:10]")

model_id = "meta-llama/Llama-3.2-1B-Instruct"

# Initialize the tokenizer and model. Adjust torch_dtype and device as needed.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

# Subclass the TextGenerationPipeline to customize preprocessing and postprocessing
class CustomTextGenerationPipeline(TextGenerationPipeline):
    def preprocess(self, inputs, **kwargs):
        """
        Preprocess the input text. Here we simply tokenize the input string.
        """
        # Tokenize input text; you can add truncation or other options if desired.
        return self.tokenizer(inputs, return_tensors="pt", truncation=True).to(self.device)
    
    def _forward(self, model_inputs, **kwargs):
        """
        Forward pass to generate text.
        """
        # Pass max_length and temperature from kwargs (or use defaults)
        max_length = kwargs.get("max_length", 1024)
        temperature = kwargs.get("temperature", 0.7)
        # Generate text using the model
        return self.model.generate(**model_inputs, max_length=max_length, temperature=temperature)
    
    def postprocess(self, model_outputs, **kwargs):
        """
        Convert model output token IDs back into a string.
        """
        generated_text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        return [{"generated_text": generated_text.strip()}]

# Instantiate our custom pipeline
custom_generate = CustomTextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda")

# Loop through a few puzzles and generate responses using our custom pipeline
for i in range(len(knight_knaves)):
    puzzle = knight_knaves[i]['quiz']
    names = knight_knaves[i]["names"]
    prompt = (
        f"Consider the following Knights and Knaves puzzle:\n\n"
        f"{puzzle}\n\n"
        f"Both {names[0]} and {names[1]} are either knights (who always tell the truth) or knaves (who always lie). "
        f"Based on the puzzle above, provide a detailed chain-of-thought reasoning to determine the role of each person and then give your final answer."
    )
    
    response = custom_generate(prompt, max_length=1024, temperature=0.7)
    print("Model Response:\n", response[0]["generated_text"])
    print("-" * 80)
