import gym
import torch
from load_policy import load_policy
from agents.model.mlp import MLPNet
from agents.algorithms.dagger import DAgger

class Config():
    seed = 3
    envname = 'Humanoid-v1'
    env = gym.make(envname)
    method = 'DA'
    device = torch.device('cpu')
    expert_path = './experts/'
    model_save_path = './models/'
    n_expert_rollouts = 30 # number of rollouts from expert
    n_dagger_rollouts = 10 # number of new rollouts from learned model for a DAgger iteration
    n_dagger_iter = 10 # number of DAgger iterations
    n_eval_rollouts = 10 # number of rollouts for evaluating a policy
    L2 = 0.00001
    lr = 0.0001
    epochs = 20
    batch_size = 64

    eval_steps = 500
    


def main():
    config = Config()
    print('*' * 20, config.envname, config.method, '*' * 20)
    env = config.env
    if config.seed:
        env.seed(config.seed)
        torch.manual_seed(config.seed)
    agent = MLPNet(env.observation_space.shape[0], env.action_space.shape[0]).to(config.device)
    expert = load_policy(config.expert_path + "Humanoid-v2" + '.pkl')
    method = config.method
    dagger = DAgger()
    agent = dagger.train(config, agent, expert)

    
    avrg_mean, avrg_std = Eval(config, expert)
    print('[expert] avrg_mean:{:.2f}  avrg_std:{:.2f}'.format(avrg_mean, avrg_std))
        
    avrg_mean, avrg_std = Eval(config, agent)
    print('[agent] avrg_mean:{:.2f}  avrg_std:{:.2f}'.format(avrg_mean, avrg_std))

if __name__ == '__main__':
    main()
    
