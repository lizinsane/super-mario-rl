import sys
import time
import os

import gym_super_mario_bros
import torch
import torch.nn as nn
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

from wrappers import SkipFrame, ProcessFrame

# ============================================================================
# PARAMETER (müssen mit ppo2.py übereinstimmen)
# ============================================================================
FRAME_STACK = 4
RESIZE_SHAPE = (84, 84)
BEST_MODEL_PATH = "mario_ppo_best.pt"
LAST_MODEL_PATH = "mario_ppo_model.pt"
EVAL_MODEL = BEST_MODEL_PATH  # Welches Modell soll evaluiert werden? 


# ============================================================================
# NEURONALES NETZWERK (Actor-Critic) - IDENTISCH ZU ppo2.py
# ============================================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic Netzwerk für PPO.
    Nimmt 4 gestackte Frames als Input und gibt Policy und Value aus.
    """
    def __init__(self, input_channels, num_actions):
        super(ActorCritic, self).__init__()
        
        # Convolutional Layers für Feature Extraction
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Berechne die Größe nach Conv-Layers
        conv_out_size = self._get_conv_out_size(input_channels)
        
        # Actor Head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Critic Head (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_out_size(self, input_channels):
        """Berechnet die Output-Größe der Conv-Layers"""
        dummy_input = torch.zeros(1, input_channels, *RESIZE_SHAPE)
        conv_out = self.conv(dummy_input)
        return conv_out.shape[1]
    
    def forward(self, x):
        """Forward Pass durch das Netzwerk"""
        x = x.float() / 255.0  # Normalisierung
        features = self.conv(x)
        return self.actor(features), self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        """
        Gibt Action, Log-Probability, Entropy und Value zurück.
        Wird für Training und Sampling verwendet.
        """
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def act(self, obs):
        """Wählt Action basierend auf der Policy (für Evaluation)"""
        logits, _ = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def make_eval_env(world=1, stage=1):
    """
    Erstellt ein Mario Environment für Evaluation (OHNE CustomReward).
    """
    env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, RESIZE_SHAPE)
    env = FrameStack(env, FRAME_STACK)
    env = ProcessFrame(env, FRAME_STACK, RESIZE_SHAPE)
    return env


# ============================================================================
# MAIN - EVALUATION
# ============================================================================

if __name__ == "__main__":
    # Standard: mario_ppo_model.pt im gleichen Verzeichnis wie eval.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(script_dir, EVAL_MODEL)
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else default_ckpt
    
    # Optional: World und Stage als Parameter
    world = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    stage = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    print("=" * 60)
    print(f"🎮 Mario PPO Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Level: World {world}-{stage}")
    print("=" * 60)
    
    # Device Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Environment erstellen
    env = make_eval_env(world, stage)
    
    # Erstelle PPO-Modell
    model = ActorCritic(input_channels=FRAME_STACK, num_actions=len(SIMPLE_MOVEMENT))
    model = model.to(device)

    # Lade Checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Checkpoint geladen (Update: {checkpoint.get('update', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Modell geladen")
    except FileNotFoundError:
        print(f"❌ Checkpoint nicht gefunden: {ckpt_path}")
        print(f"Bitte trainiere zuerst mit ppo2.py oder gib einen gültigen Pfad an.")
        sys.exit(1)
    
    model.eval()
    print("=" * 60)
    
    # Evaluation Loop
    total_score = 0.0
    done = False
    s = env.reset()
    s = torch.FloatTensor(s).unsqueeze(0).to(device)
    
    print("🎮 Starte Evaluation...\n")
    step_count = 0
    
    while not done:
        env.render()
        with torch.no_grad():
            a = model.act(s)
        s_prime, r, done, info = env.step(a)
        s_prime = torch.FloatTensor(s_prime).unsqueeze(0).to(device)
        total_score += r
        s = s_prime
        step_count += 1
        time.sleep(0.01)

    # Ergebnisse
    x_pos = info.get('x_pos', 0)
    final_world = info.get('world', world)
    final_stage = info.get('stage', stage)
    coins = info.get('coins', 0)
    flag_get = info.get('flag_get', False)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation beendet!")
    print("=" * 60)
    print(f"   Score: {info.get('score', 0)}")
    print(f"   Level: {final_world}-{final_stage}")
    print(f"   X-Position: {x_pos} pixels")
    print(f"   Münzen: {coins}")
    print(f"   Steps: {step_count}")
    print(f"   Level geschafft: {'✅ JA!' if flag_get else '❌ Nein'}")
    print("=" * 60)
    
    env.close()