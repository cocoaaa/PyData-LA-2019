import torch
import torch.optim as optim

def save_model(file, model, opt, with_opt=True):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`)"
    if opt is None:
        with_opt=False
    state = model.state_dict()
    if with_opt: state = {'model': state, 'opt':opt.state_dict()}
    torch.save(state, file)

def load_model(file, model, opt=None, device=None, strict=True) -> None:
    "Load `model` from `file` along with `opt` (if available, and if `with_opt`)"
    if isinstance(device, int): device = torch.device('cuda', device)
    elif device is None: device = 'cpu'
    state = torch.load(file, map_location=device)
    hasopt = set(state)=={'model', 'opt'}
    model_state = state['model'] if hasopt else state
    model.load_state_dict(model_state, strict=strict)
    if opt is None: return
    is_valid_opt = isinstance(opt, optim.Optimizer)
    if not is_valid_opt:
        raise ValueError('Failed to load optim satate\n'
                         f'opt is not optim.Optimizer type {opt.__class__}')
    if not hasopt:
        print("Saved state doesn't have optim state. Only loaded model state")
        return
    else:
        opt.load_state_dict(state['opt'])
        return