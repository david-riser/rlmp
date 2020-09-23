import torch


def ntd_loss(online_model, target_model, states, actions, 
             next_states, rewards, dones, gamma=0.99, n=1):
    """ Compute the n-step TD-error using pytorch 
        for a set of transitions.  This function 
        returns the loss per transition, not the 
        mean of the loss function over the batch.
    """
    q_values = online_model(states)
    next_q_values = target_model(next_states)
    q_values = q_values.gather(1, actions.view(-1,1)).squeeze(1)
    next_q_values = next_q_values.max(1).values
    qhat = (rewards + gamma**n * next_q_values * (1 - dones))
    loss = (q_values - qhat.detach()).pow(2)
    return loss
