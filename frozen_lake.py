# Frozen Lake with epsilon-greedy algotrithm

import gymnasium as gym
import numpy as np

class Agent:    
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def train(self, episodes=1000, alpha = 0.5, gamma = 0.9):
        epsilon = 1.0
        eps_dacay = epsilon / episodes
        n_success = 0
    
        for _ in range(episodes):
            state, prob = self.env.reset() 
            term = False
            trunc = False
          
            while not term and not trunc:
                rnd = np.random.random()
          
                if rnd < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])
        
                observ, reward, term, trunc, info = self.env.step(action)
                
                self.q_table[state, action] = self.q_table[state, action] + alpha * \
                        (reward + gamma * np.max(self.q_table[observ]) - self.q_table[state, action])
                
                state = observ
          
                n_success += reward

            epsilon = max(epsilon - eps_dacay, 0)
        
        return n_success / episodes * 100

    def test(self, episodes=100, get_actions=False):
        n_success = 0

        if get_actions:
            action_seq = []

        for _ in range(episodes):
            state, prob = self.env.reset() 
            term = False
            trunc = False
          
            while not term and not trunc:
                action = np.argmax(self.q_table[state])
        
                observ, reward, term, trunc, info = self.env.step(action)
                
                state = observ
          
                n_success += reward
                
                if get_actions:
                    action_seq.append(action)

        if get_actions:
            return action_seq
        else:
            return n_success / episodes * 100

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    
    agent = Agent(env)
   
    print("Training the agent...")
    success_rate_train = agent.train()
    print(f"Success rate (Training) = {success_rate_train}%")

    print("\nTesting the agent...")
    success_rate_test = agent.test()
    print(f"Success rate (Testing) = {success_rate_test}%")
    
    actions = agent.test(episodes=1, get_actions=True)
    print(f"\nActions taken to reach the goal = {actions}")
