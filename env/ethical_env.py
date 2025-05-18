import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
from collections import defaultdict, Counter

class EthicalRescueEnv(gym.Env):
    def __init__(self, max_steps=50, victims_per_agent=3):
        super(EthicalRescueEnv, self).__init__()
        self.max_steps = max_steps
        self.victims_per_agent = victims_per_agent
        self.num_agents = 40
        self.current_step = 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(victims_per_agent, 6), dtype=np.float32)
        self.action_space = spaces.Discrete(victims_per_agent)

        self.agent_input_output_history = [defaultdict(list) for _ in range(self.num_agents)]

        # Load agent locations
        agents_df = pd.read_csv("dataset/raw/agents.csv")
        self.agent_coords = {
            i: (agents_df.iloc[i]['location_x'], agents_df.iloc[i]['location_y'])
            for i in range(self.num_agents)
        }

        # Load victim pool
        victims_df = pd.read_csv("dataset/raw/victims.csv")
        self.victim_pool = victims_df.to_dict(orient='records')

    def _generate_random_victim(self):
        return random.choice(self.victim_pool)

    def _calculate_severity(self, v):
        injury_map = {'head trauma': 1.0, 'internal bleeding': 0.85, 'burn': 0.8, 'fracture': 0.6, 'minor cut': 0.3}
        bleeding_map = {'severe': 1.0, 'moderate': 0.6, 'low': 0.3, 'none': 0.0}
        consciousness_map = {'unconscious': 1.0, 'confused': 0.6, 'alert': 0.3}

        return np.clip(np.mean([
            injury_map.get(v['injury_type'], 0),
            bleeding_map.get(v['bleeding_level'], 0),
            consciousness_map.get(v['consciousness_level'], 0)
        ]), 0.0, 1.0)

    def _calculate_urgency(self, v):
        mobility_map = {'immobile': 1.0, 'semi-mobile': 0.6, 'mobile': 0.3}

        # Age score with emphasis on <10 and >=70
        def age_score(age):
            if age < 10:
                return 1.0
            elif age >= 70:
                return 0.9
            else:
                return age / 90.0  # normalized adult range

        trapped_score = 1.0 if v['trapped'] else 0.0

        return np.clip(np.mean([
            mobility_map.get(v['mobility'], 0),
            age_score(v['age']),
            trapped_score
        ]), 0.0, 1.0)

    def _calculate_reliability(self, agent_id, severity, urgency, action):
        key = (round(severity, 1), round(urgency, 1))
        history = self.agent_input_output_history[agent_id][key]
        if not history:
            return 1.0
        counts = Counter(history)
        majority_action_count = counts.most_common(1)[0][1]
        return majority_action_count / len(history)

    def reset(self):
        self.current_step = 0
        self.agent_input_output_history = [defaultdict(list) for _ in range(self.num_agents)]
        self.victims = []
        return self._get_observations()

    def _get_observations(self):
        obs = []
        self.victims = []

        for agent_index in range(self.num_agents):
            ax, ay = self.agent_coords[agent_index]
            choices = []

            for _ in range(self.victims_per_agent):
                v = self._generate_random_victim()
                severity = self._calculate_severity(v)
                urgency = self._calculate_urgency(v)

                vx, vy = v['location_x'], v['location_y']
                distance = np.sqrt((vx - ax) ** 2 + (vy - ay) ** 2)
                distance_norm = np.clip(distance / 141.42, 0, 1)

                age_norm = v['age'] / 90
                trapped = 1.0 if v['trapped'] else 0.0
                rand1 = np.random.random()
                features = [severity, urgency, age_norm, trapped, distance_norm, rand1]
                choices.append((v, features))

            random.shuffle(choices) 
            self.victims.append(choices)
            obs.append([f[1] for f in choices])


        return np.array(obs)

    def step(self, actions, confidences):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        rewards = []
        self.step_logs = []

        for i, act in enumerate(actions):
            victim_data, features = self.victims[i][act]
            severity = features[0]
            urgency = features[1]
            distance = features[4]
            transparency = confidences[i]

            reliability = self._calculate_reliability(i, severity, urgency, act)
            fairness = (severity + urgency + (1 - distance)) / 3
            ethics = 0.4 * fairness + 0.3 * transparency + 0.3 * reliability

            reward = 10 * ethics
            if reward < 0.3:
                reward -= 0.3

            self.agent_input_output_history[i][(round(severity, 1), round(urgency, 1))].append(act)
            rewards.append(reward)

            self.step_logs.append({
                "agent_id": i,
                "severity": severity,
                "urgency": urgency,
                "distance": distance,
                "fairness_score": fairness,
                "transparency_score": transparency,
                "reliability_score": reliability,
                "ethical_score": ethics,
                "reward": reward
            })

        obs = self._get_observations()
        return obs, rewards, [done] * self.num_agents, {}
