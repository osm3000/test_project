import torch

qconfig = torch.quantization.get_default_qconfig('qnnpack')
# or, set the qconfig for QAT
qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
# set the qengine to control weight packing
torch.backends.quantized.engine = 'qnnpack'

def load_model(quantized_model:bool=True):
    if quantized_model: # Load the dynamically quantized model
        # print(f'Quantized')
        model_scripted = torch.jit.load("./saved_models/int8_model.pt")
    else: # Load the full/normal model
        # print(f'Normal')
        model_scripted = torch.jit.load("./saved_models/float_model.pt")

    # print(f'model_scripted: {model_scripted}')
    return model_scripted

def human_readable_decoding(classfication_result:list):
    final_answer = []
    for item in classfication_result:
        if item == 0:
            final_answer.append("Thought")
        else:
            final_answer.append("non-Thought")

    return final_answer

def perform_classification(embedded_sentences: list, model_type:bool):
    model = load_model(quantized_model=model_type)
    # print("Perform Classification")
    classification_results = torch.argmax(torch.softmax(
        model(torch.tensor(embedded_sentences).float()), 1), 1).detach().numpy()

    human_readable_results = human_readable_decoding(classfication_result=classification_results)
    return human_readable_results
