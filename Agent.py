import collections
import os
import gym
import torch
from gym.wrappers import FrameStack, AtariPreprocessing

from ReplayBuffer import ReplayBuffer
import numpy as np
import torch.optim as optim
import time
import random
import torch.nn.functional as F
from DQN import DQN
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# TODO: atatri preprocessing is broken use other method
class Agent:
    def __init__(self):
        # using GPU if possible
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)
        self.env = gym.make('PongNoFrameskip-v4')
        self.env = AtariPreprocessing(self.env, frame_skip=4, grayscale_obs=True,scale_obs=True,grayscale_newaxis=False)
        self.env = FrameStack(self.env,num_stack=4)

        # tensorboard integration and summary writing
        self.writer = SummaryWriter('runs/atari_pong_dqn_1_continue')
        self.print_interval = 5

        # list for printing summary to terminal
        self.rewards = []
        self.epi_length = []

        # epsilon values
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon = self.final_epsilon
        self.epsilon_max_frame = 1000000

        # variables for learning
        self.number_of_actions = self.env.action_space.n
        self.learning_rate = 0.00025
        self.number_of_episode = 5000
        self.target_update_step = 10
        self.gamma = 0.99
        self.total_step = 0
        self.epi = 0

        # replay memory variables
        self.Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.replay_start_size = 1000
        self.batch_size = 32
        buffer_size_limit = 20000  # need to be set so that it does not go over the memory limit

        # save variables
        self.load = True
        self.path_to_save_file = "model/pretrained_model.pth"
        self.save_interval = 10

        # three most important things
        self.q_net = DQN(self.number_of_actions).to(self.device)
        self.q_target_net = DQN(self.number_of_actions).to(self.device)
        self.memory = ReplayBuffer(self.batch_size, buffer_size_limit)

        # uses adam optimizer and huber loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.loss = None

        # loading from save file
        if self.load:
            checkpoint = torch.load(self.path_to_save_file)
            self.q_net.load_state_dict(checkpoint['policy_net'])
            self.q_target_net.load_state_dict(checkpoint['target_net'])
            self.memory.buffer = checkpoint['replay_memory']
            self.optimizer = checkpoint['optimizer']
            self.epi = checkpoint['epi_num']

    # all the learning/training loop is in this function
    def main(self):
        while self.epi < self.number_of_episode:
            # get state when reset
            state = self.env.reset()
            state = self.np_to_tensor_with_preprocessing(state)
            cumulative_reward = 0
            step = 0
            total_loss = 0
            while True:
                self.env.render()
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)

                cumulative_reward += reward
                # self.to_image(next_state)
                next_state = self.np_to_tensor_with_preprocessing(next_state)

                # push to replay memory
                self.memory.push(state, action, torch.tensor([reward]).to(self.device), next_state)

                # when it goes past max_step or reaches goal
                if done:
                    self.rewards.append(cumulative_reward)
                    self.epi_length.append(step)
                    # writing to tensorboard
                    self.writer.add_scalar("Rewards", cumulative_reward, self.total_step)
                    self.writer.add_scalar("Episode Length", step, self.total_step)
                    self.writer.add_scalar("Epsilon", self.epsilon, self.total_step)
                    self.writer.add_scalar("Loss", total_loss, self.total_step)
                    self.writer.flush()
                    break

                # learning if possible
                if len(self.memory.buffer) > self.replay_start_size:
                    total_loss += self.learn()

                    # updates target network every few steps
                    if step % self.target_update_step == 0:

                        self.q_target_net.load_state_dict(
                            self.q_net.state_dict())
                step += 1
                self.total_step += 1
                del state
                state = next_state
                del next_state
            self.print_summary()
            self.checkpoint_save()
            self.epi+=1
        # saving model in the end
        torch.save(self.q_net, 'model/dqn_model_final.pth')
        self.env.close()
        self.writer.close()

    # deep Q learning with replay memory
    def learn(self):
        # not starting learning until sufficient data is collected
        if len(self.memory.buffer) < self.batch_size:
            return

        # get samples for learning
        mini_batch = self.memory.sample()

        # pack it to individual tensor
        batch = self.Transition(*zip(*mini_batch))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)

        # get q value from policy network
        q_values = self.q_net(state_batch).gather(1, action_batch)

        # get q value from target network using next state
        max_next_q = self.q_target_net(non_final_next_states).max(1)[0].unsqueeze(1).detach()

        target = reward_batch + self.gamma * max_next_q

        # loss is square if >1 absolute value if <=1
        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        # backpropagation
        loss.backward()
        self.optimizer.step()
        return loss

    # getting action based on epsilon-greedy
    def get_action(self, state):
        # epsilon-greedy
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.epsilon_max_frame
        rand = random.random()
        if rand > self.epsilon:
            return self.q_net(state).max(1)[1].view(1, )
        else:
            return torch.randint(0, self.number_of_actions, (1,), device=self.device)

    # changing numpy array to image for verification
    def to_image(self, state):
        y=np.transpose(np.array(state),(1,2,0))*255
        for i in range(4):
            z = y[:, :, i]
            im = Image.fromarray(np.uint8(z))
            im.show()

    # numpy array to tensor
    def np_to_tensor_with_preprocessing(self, state):

        changed = np.expand_dims(state, axis=0)
        return torch.from_numpy(changed).to(self.device)

    # saves every interval
    def checkpoint_save(self):
        if self.epi % self.save_interval == 0 and self.epi != 0:
            torch.save(
                {'policy_net': self.q_net.state_dict(), 'target_net': self.q_target_net.state_dict(),
                 'replay_memory': self.memory.buffer,
                 'optimizer': self.optimizer, 'epi_num': self.epi}, "model/pretrained_model.pth")

    # printing summary in the terminal
    def print_summary(self):
        if self.epi % self.print_interval == 0:
            print("episode: {} / step: {:.2f} / reward: {:.3f}".format(self.epi, np.mean(self.epi_length),
                                                                       np.mean(self.rewards)))
            self.epi_length = []
            self.rewards = []



if __name__ == "__main__":
    a = Agent()
    a.main()

