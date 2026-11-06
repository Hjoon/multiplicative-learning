import torch.nn.functional as F

def objective(model, x, labels, target, loss):
    if loss == "ce":
        return F.cross_entropy(model.logits(x), labels)
    if loss == "mse":
        return 0.5 * (model(x) - target).square().mean()
    raise ValueError(loss)

def accuracy(scores, labels):
    return (scores.argmax(1) == labels).float().mean().item()
