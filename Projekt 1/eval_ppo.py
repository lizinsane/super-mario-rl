import sys
import time
import os

import gym_super_mario_bros
import torch
import torch.nn as nn
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *


# PPO ActorCritic Model (same as in ppo.py)
class ActorCritic(nn.Module):
    def __init__(self, n_frame, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_frame, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(20736, 512)
        self.policy_head = nn.Linear(512, act_dim)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x.permute(2, 0, 1)
        x = self.net(x)
        x = x.reshape(-1, 20736)
        x = torch.relu(self.linear(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()


def arange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)

if __name__ == "__main__":
    # Standard: mario_1_1_ppo.pt im gleichen Verzeichnis wie eval_ppo.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(script_dir, "mario_1_1_ppo.pt")
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else default_ckpt
    
    print(f"Load PPO checkpoint from {ckpt_path}")
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Erstelle PPO-Modell
    model = ActorCritic(n_frame, env.action_space.n).to(device)

    # Lade Checkpoint (unterstützt beide Formate)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint geladen (Update: {checkpoint.get('update', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Modell geladen")
    
    model.eval()
    
    total_score = 0.0
    done = False
    s = env.reset()
    s = torch.FloatTensor(s).unsqueeze(0).to(device)
    
    print("🎮 Starte Evaluation...")
    while not done:
        env.render()
        with torch.no_grad():
            a = model.act(s)
        s_prime, r, done, info = env.step(a)
        s_prime = torch.FloatTensor(s_prime).unsqueeze(0).to(device)
        total_score += r
        s = s_prime
        time.sleep(0.01)

    stage = env.unwrapped._stage
    x_pos = info.get('x_pos', 0)
    print(f"✅ Evaluation beendet:")
    print(f"   Total Score: {total_score:.0f}")
    print(f"   Stage: {stage}")
    print(f"   X-Position: {x_pos} pixels")