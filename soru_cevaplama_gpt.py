from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch
import warnings

warnings.filterwarnings("ignore")

model_name="gpt2"

tokenizer=GPT2Tokenizer.from_pretrained(model_name)
model=GPT2LMHeadModel.from_pretrained(model_name)

def generate_answer(context,question):
    
    input_text=f"Question :{question}, context: {context}. Please answer the question according to context"
    
    # tokenize
    inputs=tokenizer.encode(input_text, return_tensors="pt")
    
    # modeli çalistir
    with torch.no_grad():
        outputs=model.generate(inputs, max_length=500)
        
    # uretilen yaniti decode edelim
    answer=tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # yanıtları ayıklayalım
    
    anser=answer.split("Answer:")[-1].strip()
    
    return answer

question="what is the capital ol france"
context="France, officiality the french Republic, is a country whose capital is Paris"

answer=generate_answer(context,question)

print(f"Answer: {answer}")
