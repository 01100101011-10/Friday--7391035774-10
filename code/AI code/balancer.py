import argparse
import gym

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Run an environment')
    parser.add_argument('--input-env', dest='input_env', required=True,
            choices=['cartpole', 'mountaincar', 'pendulum'], 
            help='Specify the name of the environment')
    return parser

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_env = args.input_env

    name_map = {'cartpole': 'CartPole-v0', 
                'mountaincar': 'MountainCar-v0',
                'pendulum': 'Pendulum-v0'}

    env = gym.make(name_map[input_env])

    for _ in range(20):
        observation = env.reset()

        for i in range(100):
            env.render()

            print(observation)

            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            
            if done:
                print('Episode finished after {} timesteps'.format(i+1))
                break
