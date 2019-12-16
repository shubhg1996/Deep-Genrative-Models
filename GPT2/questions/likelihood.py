import torch
import torch.nn as nn


def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar. 
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """

    with torch.no_grad():
        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`
        past = None
        logits, past = model(text, past=past)
        cent = nn.CrossEntropyLoss()
        loss = cent(logits[:,:-1,:].view(-1,logits.size(-1)), text[:,1:].view(-1))
        # print(loss)
        return -loss*(text.size(-1) - 1)
        # pred_logits = []
        # for pred_text in list_text[:,1:]:
        #     logits, past = model(current_text, past=past)
        #     current_logits = logits[:, -1, :]
            

            
