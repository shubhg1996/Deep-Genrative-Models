import torch
import torch.nn.functional as F
from tqdm import trange
from torch.distributions import Categorical

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def sample(model, start_text, config):
    length = config.n_ctx // 2

    current_text = start_text
    past = None
    output = [start_text]
    with torch.no_grad():
        for _ in trange(length):
            logits, past = model(current_text, past=past)
            # logits: a tensor of shape (batch_size, length_of_current_text, size_of_vocabulary)

            current_logits = logits[:, -1, :]
            logits = top_k_logits(current_logits, k=config.top_k)

            ##TODO:
            ## 1) sample using the given `logits` tensor;
            ## 2) append the sample to the list `output`;
            ## 3) update `current_text` so that sampling can continue.
            logits = F.softmax(logits)
            sampler = Categorical(logits)
            sample = sampler.sample()
            current_text = sample.unsqueeze(0)
            output.append(current_text)
            
        output = torch.cat(output, dim=1)
        return output
