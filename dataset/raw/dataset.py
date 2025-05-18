import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()

# Constants
NUM_VICTIMS = 500
NUM_AGENTS = 100
NUM_SCENARIOS = 10

# Utility functions
def random_coord():
    return round(random.uniform(0, 100), 2)

def random_choice_weighted(choices, weights):
    return random.choices(choices, weights=weights, k=1)[0]

# Generate victims
injury_types = ['fracture', 'burn', 'head trauma', 'minor cut', 'internal bleeding']
mobility_levels = ['immobile', 'semi-mobile', 'mobile']
genders = ['male', 'female', 'other']
bleeding_levels = ['none', 'low', 'moderate', 'severe']
consciousness_levels = ['alert', 'confused', 'unconscious']

victims = []
for i in range(NUM_VICTIMS):
    victims.append({
        "victim_id": f"V{i+1}",
        "location_x": random_coord(),
        "location_y": random_coord(),
        "injury_type": random_choice_weighted(injury_types, [0.2]*5),
        "mobility": random_choice_weighted(mobility_levels, [0.3, 0.4, 0.3]),
        "age": random.randint(5, 85),
        "gender": random.choice(genders),
        "trapped": random.choice([True, False]),
        "bleeding_level": random_choice_weighted(bleeding_levels, [0.1, 0.2, 0.4, 0.3]),
        "consciousness_level": random_choice_weighted(consciousness_levels, [0.5, 0.3, 0.2])
    })

victims_df = pd.DataFrame(victims)

# Generate agents
agent_types = ['drone', 'robot', 'human team']
capabilities = ['carry', 'medical', 'digging']

agents = []
for i in range(NUM_AGENTS):
    agents.append({
        "agent_id": f"A{i+1}",
        "agent_type": random.choice(agent_types),
        "capability": random.choice(capabilities),
        "location_x": random_coord(),
        "location_y": random_coord(),
        "status": random.choice(['available', 'busy']),
        "battery_level": random.randint(30, 100)
    })

agents_df = pd.DataFrame(agents)

# Generate scenarios
disaster_types = ['earthquake', 'flood', 'fire']
times_of_day = ['morning', 'afternoon', 'evening', 'night']
weather_conditions = ['clear', 'rainy', 'stormy']

scenarios = []
for i in range(NUM_SCENARIOS):
    scenarios.append({
        "scenario_id": f"S{i+1}",
        "disaster_type": random.choice(disaster_types),
        "severity_level": random.randint(1, 10),
        "time_of_day": random.choice(times_of_day),
        "weather": random.choice(weather_conditions)
    })

scenarios_df = pd.DataFrame(scenarios)

# Save to CSV
victims_path = "dataset/raw/victims.csv"
agents_path = "dataset/raw/agents.csv"
scenarios_path = "dataset/raw/scenarios.csv"

victims_df.to_csv(victims_path, index=False)
agents_df.to_csv(agents_path, index=False)
scenarios_df.to_csv(scenarios_path, index=False)

victims_path, agents_path, scenarios_path
