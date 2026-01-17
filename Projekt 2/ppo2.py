"""
PPO Training für Super Mario Bros mit parallelen Environments
"""
import csv
import os
from datetime import datetime
from collections import deque

import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

from wrappers import SkipFrame, CustomReward, ProcessFrame

# ============================================================================
# TRAININGSPARAMETER
# ============================================================================
NUM_ENVS = 8                    # Anzahl paralleler Environments
STEPS_PER_UPDATE = 256          # Schritte pro Update (insgesamt über alle Envs)
MAX_UPDATES = 1000              # Maximale Anzahl Updates pro Training
GAMMA = 0.9                     # Discount Factor
GAE_LAMBDA = 0.95               # GAE Lambda Parameter
CLIP_EPSILON = 0.2              # PPO Clipping Parameter
LEARNING_RATE = 2.5e-4          # Lernrate
PPO_EPOCHS = 4                  # Anzahl Epochen pro Update
MINIBATCH_SIZE = 64             # Minibatch Größe
VALUE_COEF = 0.5                # Value Loss Koeffizient
ENTROPY_COEF_START = 0.04       # Start Entropy Koeffizient
ENTROPY_COEF_END = 0.04         # End Entropy Koeffizient # 0.001
MAX_GRAD_NORM = 0.5             # Gradient Clipping
FRAME_STACK = 4                 # Anzahl gestackter Frames
RESIZE_SHAPE = (84, 84)         # Frame Größe

# ============================================================================
# REWARD PARAMETER
# ============================================================================
X_REWARD = 0.7                  # Belohnung pro Pixel X-Fortschritt (erhöht für mehr Motivation)
COIN_REWARD = 5.0               # Belohnung pro gesammelter Münze #10.0
DEATH_PENALTY = 30.0            # Bestrafung für Tod (reduziert, damit Risiko lohnt)
FLAG_REWARD = 1000.0            # Belohnung für Level-Abschluss (erhöht für starken Anreiz)
IDLE_PENALTY = 0.1              # Strafe pro Frame ohne X-Fortschritt (motiviert Vorwärtsdrang)

# ============================================================================
# LEVEL-KONFIGURATION
# ============================================================================
# Maximale X-Position für jedes Level (ca. Werte)
LEVEL_MAX_X = {
    (1, 1): 3266,
    (1, 2): 3266,
    (1, 3): 2514,
    (1, 4): 2430,
    (2, 1): 3266,
    (2, 2): 3266,
    (2, 3): 2514,
    (2, 4): 2430,
    # Weitere Levels können hinzugefügt werden
}

# ============================================================================
# DATEIPFADE
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "mario_ppo_model.pt")
BEST_MODEL_FILE = os.path.join(SCRIPT_DIR, "mario_ppo_best.pt")
EPISODE_LOG_FILE = os.path.join(SCRIPT_DIR, "episode_log.csv")
UPDATE_LOG_FILE = os.path.join(SCRIPT_DIR, "update_log.csv")


# ============================================================================
# NEURONALES NETZWERK (Actor-Critic)
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


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def get_cumulative_x_pos(world, stage, x_pos):
    """
    Berechnet die kumulative X-Position über alle bisherigen Stages.
    """
    cumulative = 0
    
    # Addiere alle vorherigen Stages
    for w in range(1, world + 1):
        for s in range(1, 5):
            if w == world and s == stage:
                break
            cumulative += LEVEL_MAX_X.get((w, s), 3266)
    
    # Addiere aktuelle Position
    cumulative += x_pos
    return cumulative


def make_env(world=1, stage=1):
    """
    Erstellt ein Mario Environment mit allen notwendigen Wrappern.
    """
    env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, RESIZE_SHAPE)
    env = FrameStack(env, FRAME_STACK)
    env = ProcessFrame(env, FRAME_STACK, RESIZE_SHAPE)
    env = CustomReward(env, X_REWARD, COIN_REWARD, DEATH_PENALTY, FLAG_REWARD, IDLE_PENALTY)
    return env


class ParallelEnvs:
    """
    Wrapper für parallele Environments.
    Managed mehrere Mario-Instanzen gleichzeitig.
    """
    def __init__(self, num_envs, world=1, stage=1):
        self.envs = [make_env(world, stage) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.episode_stats = [self._init_episode_stats() for _ in range(num_envs)]
        
    def _init_episode_stats(self):
        """Initialisiert Statistiken für eine Episode"""
        return {
            'score': 0,
            'x_pos': 0,
            'y_pos': 0,
            'coins': 0,
            'enemies_killed': 0,
            'time': 0,
            'max_stage': 1,
            'total_reward': 0,
            'world': 1,
            'stage': 1,
            'cumulative_x_pos': 0
        }
    
    def reset(self):
        """Setzt alle Environments zurück"""
        obs = []
        for i, env in enumerate(self.envs):
            ob = env.reset()
            obs.append(np.array(ob))
            self.episode_stats[i] = self._init_episode_stats()
        return np.array(obs)
    
    def step(self, actions):
        """
        Führt einen Step in allen Environments aus.
        Tracked Episode-Statistiken und schreibt abgeschlossene Episoden ins Log.
        """
        obs, rewards, dones, infos = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            ob, reward, done, info = env.step(action)
            
            # Update Episode Stats
            self.episode_stats[i]['total_reward'] += reward
            self.episode_stats[i]['score'] = info.get('score', 0)
            self.episode_stats[i]['x_pos'] = info.get('x_pos', 0)
            self.episode_stats[i]['y_pos'] = info.get('y_pos', 0)
            self.episode_stats[i]['coins'] = info.get('coins', 0)
            self.episode_stats[i]['time'] = info.get('time', 0)
            
            # Track max stage und world
            current_world = info.get('world', 1)
            current_stage = info.get('stage', 1)
            self.episode_stats[i]['world'] = current_world
            self.episode_stats[i]['stage'] = current_stage
            
            current_max_stage = (current_world - 1) * 4 + current_stage
            if current_max_stage > self.episode_stats[i]['max_stage']:
                self.episode_stats[i]['max_stage'] = current_max_stage
            
            # Berechne kumulative X-Position
            cumulative_x = get_cumulative_x_pos(current_world, current_stage, info.get('x_pos', 0))
            self.episode_stats[i]['cumulative_x_pos'] = cumulative_x
            
            # Bei Episode-Ende: Stats loggen
            if done:
                self._log_episode(self.episode_stats[i])
                ob = env.reset()
                self.episode_stats[i] = self._init_episode_stats()
            
            obs.append(np.array(ob))
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return np.array(obs), np.array(rewards), np.array(dones), infos
    
    def _log_episode(self, stats):
        """Schreibt Episode-Statistiken in CSV-Datei"""
        file_exists = os.path.isfile(EPISODE_LOG_FILE)
        
        with open(EPISODE_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Header schreiben falls Datei neu
            if not file_exists:
                writer.writerow([
                    'timestamp', 'max_stage', 'score', 'x_pos', 
                    'y_pos', 'coins', 'enemies_killed', 'time'
                ])
            
            # Episode-Daten schreiben
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                stats['max_stage'],
                stats['score'],
                stats['x_pos'],
                stats['y_pos'],
                stats['coins'],
                stats['enemies_killed'],
                stats['time']
            ])
    
    def close(self):
        """Schließt alle Environments"""
        for env in self.envs:
            env.close()


# ============================================================================
# PPO ALGORITHMUS
# ============================================================================

def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Berechnet Generalized Advantage Estimation (GAE).
    GAE reduziert die Varianz der Advantage-Schätzung.
    """
    advantages = []
    gae = 0
    
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[step + 1]
        
        delta = rewards[step] + gamma * next_val * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    
    return advantages


def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, 
               entropy_coef, clip_epsilon=CLIP_EPSILON, epochs=PPO_EPOCHS):
    """
    Führt PPO Update durch.
    Optimiert Policy und Value Function mit Clipped Surrogate Objective.
    """
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)
    
    # Normalisiere Advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    dataset_size = states.shape[0]
    
    for _ in range(epochs):
        # Shuffle data
        indices = np.random.permutation(dataset_size)
        
        for start in range(0, dataset_size, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            if end > dataset_size:
                continue
                
            batch_indices = indices[start:end]
            
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]
            
            # Forward pass
            _, new_log_probs, entropy, values = model.get_action_and_value(
                batch_states, batch_actions
            )
            
            # Policy Loss (Clipped Surrogate Objective)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value Loss
            value_loss = 0.5 * (batch_returns - values).pow(2).mean()
            
            # Entropy Loss (für Exploration)
            entropy_loss = -entropy.mean()
            
            # Total Loss
            loss = policy_loss + VALUE_COEF * value_loss + entropy_coef * entropy_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def log_update(update_num, avg_return, max_stage):
    "Schreibt Update-Statistiken in CSV-Datei"
    file_exists = os.path.isfile(UPDATE_LOG_FILE)
    
    with open(UPDATE_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['timestamp', 'update_nr', 'avg_return', 'max_stage'])
        
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            update_num,
            f'{avg_return:.2f}',
            max_stage
        ])


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train():
    """
    Haupttrainingsschleife für PPO.
    Sammelt Erfahrungen und führt Policy Updates durch.
    """
    print("=" * 60)
    print("PPO Training für Super Mario Bros")
    print("=" * 60)
    print(f"Anzahl paralleler Environments: {NUM_ENVS}")
    print(f"Steps pro Update: {STEPS_PER_UPDATE}")
    print(f"Maximale Updates: {MAX_UPDATES}")
    print(f"Frame Stack: {FRAME_STACK}")
    print("=" * 60)
    
    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Environment Setup
    envs = ParallelEnvs(NUM_ENVS)
    
    # Model Setup
    model = ActorCritic(input_channels=FRAME_STACK, num_actions=len(SIMPLE_MOVEMENT))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Checkpoint laden falls vorhanden
    total_updates_done = 0
    best_max_x_pos = 0  # Tracking der besten erreichten X-Position
    best_avg_return = float('-inf')  # Tracking des besten durchschnittlichen Returns
    
    if os.path.exists(CHECKPOINT_FILE):
        try:
            print(f"\nCheckpoint gefunden: {CHECKPOINT_FILE}")
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            total_updates_done = checkpoint.get('update', 0)
            best_max_x_pos = checkpoint.get('best_max_x_pos', 0)
            best_avg_return = checkpoint.get('best_avg_return', float('-inf'))
            print(f"Checkpoint geladen! Bisherige Updates: {total_updates_done}")
            print(f"   Bester Max X-Pos: {best_max_x_pos}, Bester Avg Return: {best_avg_return:.2f}")
            print(f"   Jetzt werden weitere {MAX_UPDATES} Updates durchgeführt...")
        except Exception as e:
            print(f"Fehler beim Laden des Checkpoints: {e}")
            print(f"   Training startet von vorne...")
            total_updates_done = 0
            best_max_x_pos = 0
            best_avg_return = float('-inf')
    else:
        print(f"\nKein Checkpoint gefunden. Training startet von vorne...")
    
    # Training State
    obs = envs.reset()
    global_step = 0
    
    print("\nStarte Training...\n")
    
    for update in range(1, MAX_UPDATES + 1):
        # Absolute Update-Nummer für Logging
        absolute_update = total_updates_done + update
        
        # Entropy Decay (linear basierend auf absolutem Update)
        entropy_coef = ENTROPY_COEF_START - (ENTROPY_COEF_START - ENTROPY_COEF_END) * (absolute_update / (total_updates_done + MAX_UPDATES))
        
        # Storage für Rollout
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        episode_returns = []
        max_stage_reached = 1
        
        # Sammle Erfahrungen
        steps_per_env = STEPS_PER_UPDATE // NUM_ENVS
        
        for step in range(steps_per_env):
            global_step += NUM_ENVS
            
            # Konvertiere Observation
            state = torch.FloatTensor(obs).to(device)
            
            # Wähle Actions
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(state)
            
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()
            
            # Environment Step
            next_obs, reward, done, infos = envs.step(action_np)
            
            # Speichere Daten
            states.append(obs)
            actions.append(action_np)
            log_probs.append(log_prob_np)
            rewards.append(reward)
            dones.append(done)
            values.append(value_np)
            
            # Update für nächsten Step
            obs = next_obs
            
            # Track max stage und max cumulative x_pos
            for info in infos:
                current_world = info.get('world', 1)
                current_stage = info.get('stage', 1)
                current_max = (current_world - 1) * 4 + current_stage
                max_stage_reached = max(max_stage_reached, current_max)
        
        # Berechne max cumulative x_pos im aktuellen Update
        max_cumulative_x = max([stats['cumulative_x_pos'] for stats in envs.episode_stats])
        
        # Berechne Returns mit GAE
        with torch.no_grad():
            next_state = torch.FloatTensor(obs).to(device)
            _, _, _, next_value = model.get_action_and_value(next_state)
            next_value = next_value.cpu().numpy()
        
        # Konvertiere zu Arrays
        states = np.array(states).swapaxes(0, 1).reshape(-1, FRAME_STACK, *RESIZE_SHAPE)
        actions = np.array(actions).swapaxes(0, 1).reshape(-1)
        log_probs = np.array(log_probs).swapaxes(0, 1).reshape(-1)
        rewards = np.array(rewards).swapaxes(0, 1)
        dones = np.array(dones).swapaxes(0, 1)
        values = np.array(values).swapaxes(0, 1)
        
        # Berechne GAE für jedes Environment
        all_advantages = []
        all_returns = []
        
        for env_idx in range(NUM_ENVS):
            advantages = compute_gae(
                rewards[env_idx], values[env_idx], dones[env_idx], next_value[env_idx]
            )
            returns = [adv + val for adv, val in zip(advantages, values[env_idx])]
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        
        # PPO Update
        ppo_update(
            model, optimizer, states, actions, log_probs, 
            all_returns, all_advantages, entropy_coef
        )
        
        # Logging
        avg_return = np.mean([r for r in rewards])
        log_update(absolute_update, avg_return, max_stage_reached)
        
        print(f"Update {update}/{MAX_UPDATES} (Total: {absolute_update}) | "
              f"Avg Return: {avg_return:.2f} | "
              f"Max Stage: {max_stage_reached} | "
              f"Max X-Pos: {max_cumulative_x} | "
              f"Entropy Coef: {entropy_coef:.4f}")
        
        # Prüfe ob das Modell besser ist (basierend auf max_cumulative_x)
        is_best = max_cumulative_x > best_max_x_pos
        
        if is_best:
            best_max_x_pos = max_cumulative_x
            best_avg_return = avg_return
            # Speichere Best Model
            torch.save({
                'update': absolute_update,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_max_x_pos': best_max_x_pos,
                'best_avg_return': best_avg_return,
            }, BEST_MODEL_FILE)
            print(f"  Neues BESTES Modell! X-Pos: {max_cumulative_x} -> Gespeichert: {BEST_MODEL_FILE}")
        
        # Checkpoint speichern (regelmäßig, unabhängig von Performance)
        if update % 50 == 0:
            torch.save({
                'update': absolute_update,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_max_x_pos': best_max_x_pos,
                'best_avg_return': best_avg_return,
            }, CHECKPOINT_FILE)
            print(f"  Checkpoint gespeichert: {CHECKPOINT_FILE} (Total Updates: {absolute_update})")
        
        # Prüfe ob alle Levels durchgespielt wurden (8 Welten x 4 Stages = 32)
        if max_stage_reached >= 32:
            print("\nAlle Levels durchgespielt! Training beendet.")
            break
    
    # Finales Speichern
    final_update = total_updates_done + (update if 'update' in locals() else 0)
    torch.save({
        'update': final_update,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_max_x_pos': best_max_x_pos,
        'best_avg_return': best_avg_return,
    }, CHECKPOINT_FILE)
    
    envs.close()
    print("\n" + "=" * 60)
    print("Training abgeschlossen!")
    print(f"Finale Modell gespeichert: {CHECKPOINT_FILE}")
    print(f"Bestes Modell gespeichert: {BEST_MODEL_FILE}")
    print(f"Beste Performance: X-Pos={best_max_x_pos}, Avg Return={best_avg_return:.2f}")
    print(f"Total Updates: {final_update}")
    print(f"Episode Log: {EPISODE_LOG_FILE}")
    print(f"Update Log: {UPDATE_LOG_FILE}")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train()
