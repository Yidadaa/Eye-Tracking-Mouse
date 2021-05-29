import torchvision.models as models
from types import FunctionType

def calculate_num_of_learned_params(model):
    cnt = 0
    for param in model.parameters():
        if param.requires_grad:
            cnt += param.numel()
    return cnt

def human_readable(n_params):
    if n_params >= 1e6:
        return '{:.2f} million'.format(n_params/1e6)
    if n_params >= 1e3:
        return '{:.2f} thousands'.format(n_params/1e3)
    else:
        return n_params

for model_name in dir(models):
    if model_name[0].islower():
        attr = getattr(models, model_name)
        if isinstance(attr, FunctionType):
            n_params =  calculate_num_of_learned_params(attr())
            print(model_name, '\t\t\t', human_readable(n_params))