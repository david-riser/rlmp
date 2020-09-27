import gym
import numpy as np
import random
import time
import torch
import wandb

from loss import ntd_loss
from utils import expand_transitions, Transition


class NStepTrainer:
    """ A deep Q-network training class with n-step loss. """
    def __init__(self, config, online_network, target_network, optimizer,
                 buffer, epsilon_schedule, beta_schedule, env_builder):
        self.config = config
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.buffer = buffer
        self.epsilon_schedule = epsilon_schedule
        self.beta_schedule = beta_schedule
        self.env_builder = env_builder
        self.episodic_reward = []
        self.loss = []
        self.nstep_buffer = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_episode = 0

        
    def state_transformer(self, state):
        print("ST: input", type(state))
        if len(state) > 1:
            state = np.swapaxes(state, 2, 0)
            print("ST: len() > 1", type(state), state.shape)
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 3:
            state = state.view(1, *self.online_network.obs_shape)
            print("ST: shape==3", type(state), state.shape)
        return state

    
    def action_transformer(self, action):
        action = action.detach().cpu().numpy()
        action = np.argmax(action)
        return action 


    def prime_buffer(self, env):
        """ Fill the n-step buffer each time the environment
            has been reset.
        """

        # Maybe something is in there, clear it out. 
        self.nstep_buffer = [] 

        for step in range(self.config['n_steps']):
            action = self.online_network(self.state_transformer(self.state))
            action = self.action_transformer(action)
            next_state, reward, done, info = env.step(action)

            trans = Transition(
                state=self.state, action=action,
                reward=reward, next_state=next_state, done=done,
                discounted_reward=0., nth_state=None, n=None
            )
            self.nstep_buffer.append(trans)
            self.state = next_state 
            

    def train(self):
        """ Train the online network using the n-step loss. 
        """
        env = self.env_builder()
        self.state = env.reset()
        self.prime_buffer(env)

        self.step = 0
        score = 0
        for epoch in range(self.config['n_epochs']):
            epoch_loss = []
            start_time = time.time()
            for batch in range(self.config['n_batches_per_epoch']):

                # Choose an action based on the current state and 
                # according to an epsilon-greedy policy.
                epsilon = self.epsilon_schedule.value(self.step)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.action_transformer(self.online_network(
                        self.state_transformer(self.state)
                    ))

                # Update the current state of the environment by taking
                # the action and building the current transition to be 
                # added to the n-step buffer.  These states are only added
                # to the replay buffer after a delay of n-steps.
                next_state, reward, done, info = env.step(action)
                current_trans = Transition(
                    state=self.state, action=action, 
                    next_state=next_state, reward=reward, 
                    discounted_reward=None, nth_state=None,
                    done=done, n=None
                )

                # Now use the contents of the n-step buffer to construct
                # the delayed transition and add that to the prioritized
                # replay buffer to be sampled for learning.
                (delayed_states, delayed_actions, delayed_rewards, 
                 delayed_next_states, delayed_discounted_rewards, 
                 delayed_nth_states, delayed_dones, delayed_ns) = expand_transitions(
                     self.nstep_buffer, torchify=False
                 )

                # Ensure that if the current episode has ended the last
                # few transitions get added correctly to the buffer. 
                if not current_trans.done:
                    delayed_trans = Transition(
                        state=delayed_states[0], action=delayed_actions[0],
                        reward=delayed_rewards[0], next_state=delayed_next_states[0],
                        discounted_reward=np.sum([reward * self.config['gamma'] ** i for i, reward in enumerate(delayed_rewards)]),
                        nth_state=self.state, done=done, n=self.config['n_steps']
                    )
                    self.buffer.add(delayed_trans)

                else:
                    for i in range(self.config['n_steps']):
                        delayed_trans = Transition(
                            state=delayed_states[i], action=delayed_actions[i],
                            reward=delayed_rewards[i], next_state=delayed_next_states[i],
                            discounted_reward=np.sum([reward * self.config['gamma'] ** j for j, reward in enumerate(delayed_rewards[i:])]),
                            nth_state=self.state, done=done, n=self.config['n_steps'] - i
                        )
                        self.buffer.add(delayed_trans)

                        
                    
                # Now that we have used the buffer, we can add the current
                # transition to the queue.  Update the current state of the 
                # environment.
                self.nstep_buffer.append(current_trans)
                if len(self.nstep_buffer) > self.config['n_steps']:
                    _ = self.nstep_buffer.pop(0)
                self.state = next_state

                if len(self.buffer) >= self.config['batch_size']:
                    # Sample a batch of experience from the replay buffer and 
                    # train with the n-step TD loss.
                    beta = self.beta_schedule.value(self.step)
                    transitions, weights, indices = self.buffer.sample(
                        self.config['batch_size'], beta)
                    (states, actions, rewards, next_states, discounted_rewards,
                     nth_states, dones, ns) = expand_transitions(transitions)
                    
                    # Calculate the loss per transition.  This is not 
                    # aggregated so that we can make the importance sampling
                    # correction to the loss.
                    #
                    # First we calculate the loss for 1-step ahead, then if 
                    # required, we look ahead n-steps and add that to our loss.
                    # Importance sampling weights are based on the 1-step loss.
                    loss = ntd_loss(
                        online_model=self.online_network, 
                        target_model=self.target_network,
                        states=states, actions=actions,
                        next_states=next_states, rewards=rewards,
                        dones=dones, gamma=0.99, n=1
                    )       
                    weights = torch.FloatTensor(weights).to(self.device)
                    loss = loss * weights
                    priorities = loss + 1e-5
                    priorities = priorities.detach().cpu().numpy()
                    self.buffer.update_priorities(priorities, indices)
                    loss = loss.mean()
                    
                    if self.config['n_steps'] > 1:
                        nstep_loss = ntd_loss(
                            online_model=self.online_network, 
                            target_model=self.target_network,
                            states=states, actions=actions,
                            next_states=nth_states, rewards=discounted_rewards,
                            dones=dones, gamma=0.99, n=ns
                        )
                        nstep_loss = nstep_loss.mean()
                        loss += nstep_loss  

                    # Take the step of updating online network parameters
                    # based on this batch loss.
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # End of training step actions
                    epoch_loss.append(loss.detach().cpu().numpy())
                
                # End of every step actions
                if self.step % self.config['update_interval'] == 0:
                    self.target_network.load_state_dict(self.online_network.state_dict())
                self.step += 1 
                score += current_trans.reward

                if current_trans.done:
                    if score > self.best_episode:
                        self.best_episode = score

                    self.episodic_reward.append(score)
                    score = 0
                    self.state = env.reset()
                    self.prime_buffer(env)

                    wandb.log({"episodic_reward":self.episodic_reward[-1]})
                    
            # End of batch actions
            self.loss.append(np.mean(epoch_loss))
            print("Epoch {0}, Score {1:6.4f}, Loss {2:6.4f}, Time {3:6.4f}".format(
                epoch, score, self.loss[-1], time.time() - start_time
            ))

            wandb.log({
                "time":time.time() - start_time,
                "loss":self.loss[-1]
            })

        wandb.log({"best_episode":self.best_episode})
