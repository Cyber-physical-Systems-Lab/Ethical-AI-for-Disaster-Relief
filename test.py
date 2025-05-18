import torch
import numpy as np
import pandas as pd
from mappo.network import Actor
from env.ethical_env import EthicalRescueEnv

# ---------- CONFIG ----------
MODEL_PATH = "model/seed_6287/final_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Sample Victims (Each with 6 features) ----------
sample_victims = np.array([
     # High severity/urgency, moderate distance
    [0.3, 0.2, 0.1, 0.0, 0.8, 0.0],  # Low priority
    [0.6, 0.9, 0.4, 0.0, 0.4, 0.0],
    [0.7, 0.6, 0.3, 1.0, 0.5, 0.0],    # Urgent, fair distance
], dtype=np.float32)

# ---------- Flatten input and convert to tensor ----------
obs_tensor = torch.tensor(sample_victims.reshape(1, -1), dtype=torch.float32).to(DEVICE)

# ---------- Load Trained Model ----------
actor = Actor(obs_dim=18, act_dim=3).to(DEVICE)
actor.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
actor.eval()

# ---------- Model Prediction ----------
with torch.no_grad():
    logits, value = actor(obs_tensor)  # Unpack logits and critic value
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    logp = dist.log_prob(action)
    confidence = probs[0, action.item()].item()

selected = action.item()
print(f"\n🧠 Model selected victim index: {selected}")

# ---------- Evaluate Ethical Scores ----------
env = EthicalRescueEnv(max_steps=1, victims_per_agent=3)
logs = []

victim_features = sample_victims
ethical_scores = []
transparency = confidence  # same for all victims

for idx in range(3):
    features = victim_features[idx]
    severity = features[0]
    urgency = features[1]
    distance = features[4]

    reliability = env._calculate_reliability(0, severity, urgency, idx)  # ✅ Correct
    fairness = (severity + urgency + (1 - distance)) / 3
    ethical_score = 0.4 * fairness + 0.3 * transparency + 0.3 * reliability
    ethical_scores.append(ethical_score)

    logs.append({
        "victim_index": idx,
        "fairness": round(fairness, 3),
        "transparency": round(transparency, 3),
        "reliability": round(reliability, 3),
        "ethical_score": round(ethical_score, 3),
        "chosen": idx == selected
    })

# ---------- Print Table ----------
df = pd.DataFrame(logs)
print("\n📊 Ethical Evaluation:")
print(df)

best_idx = int(np.argmax(ethical_scores))
print(f"\n✅ Was the choice optimal? {'Yes' if selected == best_idx else 'No'} (Best index: {best_idx})")
