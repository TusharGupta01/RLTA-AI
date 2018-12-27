import gym
import torch
import numpy as np
import pydash as ps

class DAgger:
	def __init__(self):
		"""
			Initialize Dataset (expert)
			Initialize π1 to any policy in π.
			for i = 1 to N do
				Let πi = ßiπ* + (1 - ßi)πi.
				Sample T-step trajectories using πi.
				Get dataset Di = {(s, π*(s))} of visited states by πi and actions given by expert.
				Aggregate datasets: D <- D U Di.
				Train classifier πi+1 on D (or use online learner to get πi+1 given new data Di).
			end for
			Return best ˆ⇡i on validation.
		"""
		pass

	def run(self, config, agent, expert):
		expert_obs, expert_actions, *_ = run_agent(config, expert, config.n_expert_rollouts)
		expert_obs = torch.from_numpy(expert_obs).to(config.device)
		expert_actions = torch.from_numpy(expert_actions).to(config.device)
		dataset = TensorDataset(expert_obs, expert_actions)

		for k in range(config.n_dagger_iter):
			# training agent
			fit_dataset(config, agent, dataset, config.epochs)
			
			# run agent to get new on-policy observations
			new_obs, *_ = run_agent(config, agent_wapper(config, agent), config.n_dagger_rollouts)
			expert_actions = expert(new_obs)
			
			new_obs = torch.from_numpy(new_obs).to(config.device)
			expert_actions = torch.from_numpy(expert_actions).to(config.device)
			new_data = TensorDataset(new_obs, expert_actions)
			
			# add new data to dataset
			dataset = ConcatDataset([dataset, new_data])

			avg_mean, avg_std = Eval(config, agent_wapper(config, agent))
			print('[DAgger iter {}] r_mean: {:.2f}  r_std: {:.2f}'.format(k + 1, avg_mean, avg_std))

			
		return agent_wapper(config, agent)

	def run_agent(self, config, agent, num_rollouts):
		env = config.env
		max_steps = env.spec.timestep_limit

		returns = []
		observations = []
		actions = []
		for _ in range(num_rollouts):
			config.env.render()
			obs = env.reset()
			done = False
			reward = 0
			steps = 0
			while not done or max_steps >= steps:
				action = agent(obs[None, :]).reshape(-1)
				observations.append(obs)
				actions.append(action)
				obs, r, done, _ = env.step(action)
				reward += r
				steps += 1
			returns.append(reward)

		avg_mean, avg_std = np.mean(returns), np.std(returns)
		observations = np.array(observations).astype(np.float32)
		actions = np.array(actions).astype(np.float32)

		return observations, actions, avg_mean, avg_std
