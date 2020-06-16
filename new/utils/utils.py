import torch

def denorm(img):
    """ Denormalize input image tensor. (From [0,1] -> [-1,1]) 
    
    Args:
        img: input image tensor.
    """
	
    output = img / 2 + 0.5
    return output.clamp(0, 1)


def save_model(model, optimizer, file_path):
    """ Save model checkpoints. """

    state = {'model' : model.state_dict(),
             'optim' : optimizer.state_dict(),
            }
    torch.save(state, file_path)
    return

def load_model(model, optimizer, file_path):
    """ Load previous checkpoints. """

    prev_state = torch.load(file_path)
    
    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])

    return model, optimizer