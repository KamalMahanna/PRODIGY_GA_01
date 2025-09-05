import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(input_text):
    # Load the tokenizer and model from the saved directory
    model_path = 'model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Prepare the input text you want to generate predictions for
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=700, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', type=str, required=True)
    args = parser.parse_args()

    print(main(args.input_text))
