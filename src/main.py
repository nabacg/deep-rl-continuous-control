import argparse
import os
from dqn_agent import Agent, train_agent, test_agent
from unityagents import UnityEnvironment




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', default="Banana.app", help="Path to Unity Environment file, allows to change which env is created. Defaults to Banana.app")
    parser.add_argument('--mode', choices=["train", "test"], default="train", help="Allows switch between training new DQN Agent or test pretrained Agent by loading model weights from file")
    parser.add_argument('--episodes', type=int, default=2000, help="Select how many episodes should training run for. Should be multiple of 100 or mean target score calculation won't make much sense")
    parser.add_argument('--eps_decay', type=float, default=0.995, help="Epsilon decay parameter value")
    parser.add_argument('--eps_end', type=float, default=0.995, help="Epsilon end value, after achieving which epsilon decay stops")
    parser.add_argument('--target_score', type=float, default=13.0, help="Target traning score, when mean score over 100 episodes ")
    parser.add_argument('--input_weights', type=str, default=None, help="Path to Q Network model weights to load into DQN Agent before training or testing")
    parser.add_argument('--output_weights', type=str, default="new_qnetwork_model_weights.pth", help="Path to save Q Network model weights after training.")
    

    args = parser.parse_args()

    train_mode = args.mode == "train"
    

    env = UnityEnvironment(file_name=args.env_file)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    print("Resetting env to {} mode".format(args.mode))
    env_info = env.reset(train_mode=train_mode)[brain_name]

    # # number of agents in the environment
    # print('Number of agents:', len(env_info.agents))

    # # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # # examine the state space
    state = env_info.vector_observations[0]
    # print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    if train_mode:
        input_weights = args.input_weights if args.input_weights and os.path.isfile(args.input_weights) else None
        train_agent(agent, env, args.output_weights, args.target_score, args.episodes, args.eps_decay, args.eps_end, input_weights)
    elif args.input_weights and os.path.isfile(args.input_weights):
        test_agent(agent, env, args.input_weights, args.episodes)
    else:
        print("Test mode requires providing input_weights path to existing file! ")
