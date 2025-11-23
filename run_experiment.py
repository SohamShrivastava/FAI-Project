import os
import argparse
from environment import SnakeEnvironment
from agents import RandomAgent, SarsaAgent, DQNAgent

def run(agent_type, max_games):
    print(f"starting experiment: {agent_type}")
    
    # pick agent 
    if agent_type == 'DQN':
        bot = DQNAgent()
    elif agent_type == 'SARSA':
        bot = SarsaAgent()
    elif agent_type == 'RANDOM':
        bot = RandomAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    env = SnakeEnvironment()

    filename = f"results_{agent_type}.csv"
    
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("episode,score\n")
    
    best_score = 0
    
    # Training Loop
    while bot.n_games < max_games:
        state_curr = bot.get_state(env)

        # action
        action = bot.select_action(state_curr)

        # step
        reward, done, score = env.step(action)
        state_next = bot.get_state(env)

        #update agent
        if agent_type == 'SARSA':
            # SARSA needs the next action for the update rule
            next_action = bot.select_action(state_next)
            bot.train_step(state_curr, action, reward, state_next, next_action, done)
        
        elif agent_type == 'DQN':
            # DQN trains on the immediate step
            bot.train_short(state_curr, action, reward, state_next, done)
            bot.remember(state_curr, action, reward, state_next, done)

        #end of episode
        if done:
            env.reset_env()
            bot.n_games += 1
            
            if agent_type == 'DQN':
                bot.train_long() # Replay memory training
            if score > best_score:
                best_score = score
                if agent_type == 'DQN':
                    bot.net.save_checkpoint()

            print(f"[{agent_type}] Episode {bot.n_games}/{max_games} | Score: {score} | Best: {best_score}")
            
            # log data to CSV
            with open(filename, "a") as f:
                f.write(f"{bot.n_games},{score}\n")

if __name__ == '__main__':
    # command line args
    parser = argparse.ArgumentParser(description="Run Snake AI Experiments")
    
    parser.add_argument('--agent', type=str, default='DQN', 
                        choices=['RANDOM', 'SARSA', 'DQN'],
                        help='agent to train (RANDOM, SARSA, DQN)')
    
    parser.add_argument('--episodes', type=int, default=500, 
                        help='number of games to play')

    args = parser.parse_args()
    
    # run
    run(args.agent, args.episodes)