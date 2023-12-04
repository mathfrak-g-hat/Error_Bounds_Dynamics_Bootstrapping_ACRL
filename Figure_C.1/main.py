### Borrowed from: https://github.com/sfujim/TD3/tree/master
import numpy as np
import torch
import utils
import gym
import os
from datetime import datetime
import matplotlib.pyplot as plt
import TD3

# Runs a set of test episodes and computes the average return based on the OpenAI reward
def eval_agent(agent, eval_env, n_eval_episodes = 5):
    
    #eval_env.reset(seed = seed)  # use the same set of initial states for each evaluation
    avg_return = 0.0
    for _ in range(n_eval_episodes):
        state = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_return += reward
    avg_return /= n_eval_episodes
    print("Return = {:.1f}, Q_nse = {:.1f} after {:d} timesteps".format(avg_return, Q_nse, t + 1))
    return avg_return



agent_name = "QH"
env_name = "Ant-v4"  # "Walker2d-v4"  # "Hopper-v4" "HalfCheetah-v4"  # "Ant-v4"  
max_timesteps = 3e6
start_timesteps = 1e4  #1*50000  # use a random policy for this many timesteps



task_name = 'walk'
#task_name = 'run'
#task_name = 'hop'

save_model = False  # True *****

tau = 0.005  #0.005
eval_interval = 5e3  #5e3  # evaluate policy at intervals of this many timesteps
policy_noise = 0.2  #0.2
expl_noise = 0.1  #0.1  # std of Gaussian exploration noise
batch_size = 256

for experiment in range(5):  # range(n) means 0 to n-1 inclusive



    seed = 11 + experiment
    Q_nse = 0.0
    
    now_string = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = "{}_{}_seed{}_{}".format(agent_name, env_name[:-3], seed, now_string)
    print("-----------------------------------------------")
    print("Agent {}, Env {}, Task {}, Seed {}".format(agent_name, env_name, task_name, seed))
    print("-----------------------------------------------")
    
    if not os.path.exists("./results"):
        os.makedirs("./results")    
    
    if save_model and not os.path.exists("./models"):
        os.makedirs("./models")
        
    env = gym.make(env_name)  #, use_contact_forces = True)
    eval_env = gym.make(env_name)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed = seed)
    env.action_space.np_random.seed(seed)
    eval_env.reset(seed = seed)
    eval_env.action_space.np_random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])
    
    agent = TD3.Agent(state_dim, action_dim, max_action, discount = 0.99,
                     tau = 0.005, policy_noise = 0.2, noise_clip = 0.5,
                     policy_freq = 2)   # 2 *****
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained agent
    t = -1
    return_record = [eval_agent(agent, eval_env)]
    Q_nse_record = [Q_nse]
    
    state = env.reset()
    done = False
    episode_timestep = 0
    episode_num = 1
    
    for t in range(int(max_timesteps)):
        
        episode_timestep += 1
        
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (agent.select_action(np.array(state))
                      + expl_noise*np.random.randn(action_dim)
                     ).clip(min_action, max_action)  # *****
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timestep < env._max_episode_steps else 0  # *****
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        Q_nse = agent.train(replay_buffer, batch_size)
       
        if done:
            state = env.reset()
            done = False
            episode_num += 1
            episode_timestep = 0
     
        # Evaluate episode
        if (t + 1) % eval_interval == 0:
            
            result_file = "./results/{}".format(file_name)
            return_record.append(eval_agent(agent, eval_env))
            Q_nse = Q_nse.detach().cpu().numpy()
            Q_nse_record.append(Q_nse)
            np.savez(result_file, return_record = return_record, Q_nse_record = Q_nse_record)

            if save_model:
                agent.save("./models/{}".format(file_name))
                 
            save_path = "./results/{}_{}_seed{}_{}.png".format(agent_name, env_name[:-3], seed, now_string)
            fig = plt.figure()
            plt.grid(which = 'both', axis = 'both')
            plt.title(env_name[:-3])
            plt.xlabel("million timesteps")
            plt.ylabel("return")
            x_pt = np.arange(len(return_record))/200.
            plt.plot(x_pt, return_record)
            fig.savefig(save_path)
            plt.close()


