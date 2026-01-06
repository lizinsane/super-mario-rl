"""
Proximal Policy Optimization (PPO) für Super Mario Bros

TRAINING OUTPUT ERKLÄRUNG:
==========================

🎮 ENVIRONMENT INFO (während des Rollouts):
--------------------------------------------
Wenn ein Environment fertig ist, wird ausgegeben:
"Env {i} done. Resetting. (info: {...})"

Die info-Dictionary enthält folgende Werte:

• coins:         Anzahl gesammelter Münzen in dieser Episode (0-100+)
                 Mehr Münzen = bessere Exploration und Punktzahl

• flag_get:      Hat Mario die Zielflagge erreicht? (True/False)
                 True = Level erfolgreich abgeschlossen! 🎯
                 False = Mario ist gestorben oder Zeit abgelaufen

• life:          Verbleibende Leben (normalerweise 2, da Spiel mit 3 startet)
                 Bei 0 = Game Over

• score:         Aktuelle Punktzahl im Spiel (0-999999)
                 Punkte durch: Münzen, Gegner besiegen, Power-Ups, Zielflagge
                 Höhere Position an Flagge = mehr Bonuspunkte

• stage:         Aktueller Level/Stage (1-4 für World 1)
                 1 = Level 1-1, 2 = Level 1-2, etc.
                 Stage > 1 bedeutet Mario hat ein Level geschafft!

• status:        Marios aktueller Status-String
                 "small" = Kleiner Mario (1 Hit = Tod)
                 "tall"  = Großer Mario (1 Hit = wird klein)
                 "fireball" = Feuer-Mario (kann Feuerbälle werfen)

• time:          Verbleibende Zeit auf der Level-Uhr (0-400)
                 Startet bei ~400, zählt runter
                 Bei 0 = Mario stirbt (Time Over)

• world:         Aktuelle Welt (normalerweise 1 für World 1)
                 SuperMarioBros-v0 spielt nur World 1

• x_pos:         Horizontale Position von Mario in Pixeln (0-3000+)
                 Höhere Werte = weiter rechts im Level
                 Level-Ende bei ca. 3161 Pixel
                 Wichtig zur Messung des Fortschritts!

• y_pos:         Vertikale Position von Mario in Pixeln
                 ~79 = auf dem Boden
                 <79 = in der Luft (springt)
                 >79 = in einer Grube (meist Tod)

• terminal_observation (done): 
                 Boolean-Wert, ob die Episode beendet ist (True/False)
                 Episode endet wenn:
                 - Mario stirbt (Gegner-Kontakt, Grube, Zeit abgelaufen)
                 - Level erfolgreich abgeschlossen (Flagge erreicht)
                 - Game Over (keine Leben mehr)
                 
                 True = Episode beendet → Environment wird automatisch resettet
                 False = Episode läuft weiter
                 
                 Wichtig: Bei done=True werden die finalen Werte (x_pos, score, etc.)
                 als letzte Observation zurückgegeben, bevor Reset erfolgt


📊 TRAINING UPDATES (alle paar Schritte):
------------------------------------------
Format: "Update {n}: avg return = {x} max_stage={y}"

• Update:        Anzahl der durchgeführten Training-Updates
                 1 Update = 1 Rollout (128 Steps × 8 Environments) + 4 Epochen Training
                 
• avg return:    Durchschnittliche Gesamtbelohnung über alle 8 Environments
                 im aktuellen Rollout
                 - Negativ (-500 bis 0): Mario kommt nicht weit, stirbt früh
                 - 0-200: Lernt Grundlagen, überlebt länger
                 - 200-500: Macht guten Fortschritt, vermeidet Gefahren
                 - >500: Sehr gute Performance, erreicht späte Level-Abschnitte
                 - >1000: Excellent! Nahe am Level-Ende
                 
• max_stage:     Höchster erreichter Stage in diesem Rollout
                 1 = Noch in 1-1
                 2 = Hat Level 1-1 geschafft! 🎉


🏆 EVALUATION (alle 10 Updates):
---------------------------------
Format: "[Eval] Update {n}: avg return = {x} info: {...}"

• avg return:    Durchschnittliche Belohnung über Evaluations-Episoden
                 (greedy policy, keine Exploration)
                 Zeigt die "echte" Performance des Agenten
                 
• info:          Detaillierte Info der letzten Evaluation-Episode
                 (siehe "ENVIRONMENT INFO" oben)
                 + action_count: Counter der verwendeten Aktionen
                   Zeigt welche Aktionen wie oft ausgeführt wurden
                   Hilfreich um zu sehen ob Agent diverse Aktionen nutzt

• eval_max_stage: Höchster erreichter Stage während Evaluation
                  Wenn >1: Training stoppt automatisch (Level geschafft!)


� BEISPIEL-OUTPUT ERKLÄRT:
----------------------------
"Update 10: avg return = 12.28 max_stage=1"

• Update 10:     10. Training-Update seit Start
                 = 10 × (128 Steps × 8 Envs) = 10.240 Spielschritte erlebt
                 Ein Update = 1 Rollout sammeln + 4 Epochen trainieren
                 
• avg return = 12.28:
                 Durchschnittliche Belohnung über alle 8 Environments
                 
                 INTERPRETATION DER WERTE:
                 < 0:        Mario stirbt sofort, sehr schlecht
                 0-50:       Frühe Lernphase, lernt Grundlagen ← 12.28 ist hier!
                 50-200:     Macht Fortschritte, vermeidet Gegner besser
                 200-500:    Gute Performance, kommt weit im Level
                 500-1000:   Sehr gut, erreicht späte Abschnitte
                 1000+:      Exzellent, nahe am Levelende
                 
                 Bei 12.28 bedeutet das:
                 ✅ Mario überlebt länger als zu Beginn
                 ✅ Bewegt sich vorwärts (positiver Score!)
                 ✅ Lernt grundlegende Steuerung
                 ⚠️ Stirbt aber noch oft früh
                 ⚠️ Keine komplexen Strategien
                 
• max_stage=1:   Höchster erreichter Stage in diesem Rollout
                 Stage 1 = Level 1-1 (noch nicht abgeschlossen)
                 Stage 2 = Level 1-2 (Mario hat 1-1 geschafft! 🎉)
                 
                 Bei max_stage=1:
                 ❌ Level noch nicht geschafft
                 🎯 Training läuft weiter
                 💡 Ziel: max_stage=2 erreichen!

TYPISCHER LERNVERLAUF:
Update 1-20:     avg return -50 bis 50    → Lernt Basics
Update 20-50:    avg return 50 bis 150    → Erste Erfolge  
Update 50-100:   avg return 150 bis 400   → Gute Fortschritte
Update 100-200:  avg return 400 bis 800   → Wird richtig gut
Update 200+:     avg return 800+, stage=2 → Schafft Level! 🎉


�💡 TRAINING-VERLAUF INTERPRETIEREN:
------------------------------------
Gutes Zeichen:
  ✅ avg return steigt kontinuierlich
  ✅ x_pos Werte werden größer (Mario kommt weiter)
  ✅ max_stage erreicht 2 (Level geschafft!)
  ✅ flag_get = True in info

Warnsignal:
  ⚠️  avg return bleibt konstant negativ
  ⚠️  x_pos stagniert bei niedrigen Werten
  ⚠️  time läuft oft auf 0 (zu langsam)
  ⚠️  life = 0 sehr häufig (stirbt zu oft)

Erfolg:
  🎉 eval_max_stage > 1 → Training stoppt, Level geschafft!
  🎉 Model wird als "mario_1_1_clear.pt" gespeichert


⚙️  WICHTIGE TRAININGS-PARAMETER:
==================================

ALGORITHMUS-PARAMETER (PPO-spezifisch):
----------------------------------------
• lr (Learning Rate):         2.5e-4 (0.00025)
                              Schrittgröße für Gewichtsaktualisierungen
                              Zu hoch: Instabiles Training, oszilliert
                              Zu niedrig: Langsame Konvergenz
                              2.5e-4 ist Standard für PPO

• rollout_steps:              128
                              Anzahl der Steps pro Environment vor einem Update
                              128 Steps × 8 Envs = 1024 Samples pro Rollout
                              Mehr = stabilere Gradienten, aber länger bis Update
                              
• epochs:                     4
                              Wie oft dieselben Daten für Training verwendet werden
                              PPO nutzt Daten mehrfach (anders als Policy Gradient)
                              Zu viel: Überanpassung, Policy wird zu gierig
                              4 ist typischer PPO-Wert

• minibatch_size:             64
                              Batch-Größe für jedes Gradient-Update
                              Aus 1024 Samples werden 16 Minibatches á 64 gebildet
                              Größer = stabilere Gradienten, mehr GPU-Speicher
                              Kleiner = mehr Updates, weniger Speicher

• clip_eps (ε):               0.2
                              PPO Clipping-Parameter (kritisch für PPO!)
                              Begrenzt wie stark die Policy sich ändern darf
                              Policy Ratio wird auf [0.8, 1.2] geclipped
                              Verhindert zu große Policy-Updates
                              0.2 = Standard, 0.1-0.3 sind üblich

• vf_coef (Value Function):   0.5
                              Gewichtung des Value-Loss in der Gesamt-Loss
                              Total Loss = Policy Loss + 0.5 × Value Loss - 0.01 × Entropy
                              Höher = Value-Function wird genauer, Policy langsamer

• ent_coef (Entropy):         0.01
                              Gewichtung der Entropy-Bonus (fördert Exploration)
                              Höher = mehr Exploration, mehr Zufälligkeit
                              Zu hoch: Agent bleibt zu zufällig
                              Zu niedrig: Agent wird zu schnell gierig (greedy)

• gamma (Discount):           0.99
                              Wie stark zukünftige Belohnungen gewichtet werden
                              0.99 = 99% der zukünftigen Belohnung zählt
                              Höher = weitsichtiger, plant langfristig
                              
• lambda (λ, GAE):            0.95
                              GAE (Generalized Advantage Estimation) Parameter
                              Trade-off zwischen Bias und Varianz
                              0.95 = Standard, balanciert Genauigkeit und Stabilität


ENVIRONMENT-PARAMETER:
----------------------
• num_env:                    8
                              Anzahl parallel laufender Environments
                              Mehr = schnellere Datensammlung, bessere GPU-Nutzung
                              Begrenzt durch RAM und GPU-Speicher
                              8 ist guter Kompromiss für Consumer-Hardware

• obs_dim (n_frame):          4
                              Anzahl gestapelter Frames als Observation
                              4 aufeinanderfolgende Frames → Agent sieht Bewegung
                              Wichtig da ein Frame allein keine Geschwindigkeit zeigt

• act_dim:                    12 (COMPLEX_MOVEMENT)
                              Anzahl möglicher Aktionen
                              12 = Kombinationen von: rechts, links, springen, etc.


SPEICHER- & EVALUATIONS-PARAMETER:
-----------------------------------
• Save Interval:              Alle 50 Updates → "mario_1_1_ppo.pt"
                              Regelmäßige Checkpoints für Fortschritt
                              
• Eval Interval:              Alle 10 Updates
                              Testet Agent ohne Exploration (greedy policy)
                              Zeigt echte Performance
                              
• Success Criterion:          eval_max_stage > 1
                              Stoppt Training wenn Level geschafft
                              Speichert finales Model als "mario_1_1_clear.pt"


NETZWERK-ARCHITEKTUR (ActorCritic):
------------------------------------
• Conv Layer 1:               4 → 32 Filter, Kernel 8×8, Stride 4
                              Extrahiert grobe Features aus Frames
                              
• Conv Layer 2:               32 → 64 Filter, Kernel 3×3, Stride 1
                              Verfeinert Features
                              
• Linear Layer:               20736 → 512 Neuronen
                              Fully-Connected Layer nach Flatten
                              
• Policy Head:                512 → 12 (Aktionen)
                              Gibt Wahrscheinlichkeit für jede Aktion
                              
• Value Head:                 512 → 1 (State Value)
                              Schätzt Wert des aktuellen Zustands


💡 PARAMETER-TUNING TIPPS:
---------------------------
Für schnelleres Training:
  → Erhöhe num_env (z.B. 16, wenn genug RAM/GPU)
  → Erhöhe rollout_steps (z.B. 256)
  
Für stabileres Training:
  → Reduziere lr auf 1e-4
  → Reduziere clip_eps auf 0.1
  
Für mehr Exploration:
  → Erhöhe ent_coef auf 0.02-0.05
  
Bei Überanpassung:
  → Reduziere epochs auf 3
  → Erhöhe Entropy Bonus
"""

from collections import Counter
import csv
import os
from datetime import datetime

import gym_super_mario_bros
import gym as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


def init_csv_logger(filename="training_log.csv"):
    """
    Initialisiert CSV-Datei für Training-Logs.
    Wenn Datei existiert, wird weiter angehängt.
    """
    file_exists = os.path.exists(filename)
    
    if not file_exists:
        # Erstelle neue CSV mit Header
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'update',
                'avg_return',
                'max_stage',
                'eval_avg_return',
                'eval_max_stage',
                'eval_coins',
                'eval_flag_get',
                'eval_life',
                'eval_score',
                'eval_status',
                'eval_time',
                'eval_x_pos',
                'eval_y_pos',
                'device'
            ])
        print(f"📊 CSV-Logger initialisiert: {filename}")
    else:
        print(f"📊 CSV-Logger wird fortgesetzt: {filename}")
    
    return filename


def log_to_csv(filename, update, avg_return, max_stage, eval_data=None):
    """
    Schreibt Training-Daten in CSV-Datei.
    
    Args:
        filename: CSV-Dateiname
        update: Update-Nummer
        avg_return: Durchschnittliche Belohnung
        max_stage: Höchster erreichter Stage
        eval_data: Optional - Dictionary mit Evaluations-Daten (avg_score, info, eval_max_stage)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Basis-Daten (bei jedem Update)
    row = [
        timestamp,
        update,
        f"{avg_return:.2f}",
        max_stage,
    ]
    
    # Evaluations-Daten (nur alle 10 Updates)
    if eval_data:
        info = eval_data.get('info', {})
        row.extend([
            f"{eval_data.get('avg_score', ''):.2f}" if eval_data.get('avg_score') else '',
            eval_data.get('eval_max_stage', ''),
            info.get('coins', ''),
            info.get('flag_get', ''),
            info.get('life', ''),
            info.get('score', ''),
            info.get('status', ''),
            info.get('time', ''),
            info.get('x_pos', ''),
            info.get('y_pos', ''),
        ])
    else:
        # Leere Felder wenn keine Evaluation
        row.extend([''] * 10)
    
    row.append(device)
    
    # Schreibe in CSV
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    return env


def get_reward(r):
    r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
    return r


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
        logprob = dist.log_prob(action)
        return action, logprob, value


def compute_gae_batch(rewards, values, dones, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=device)

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


def rollout_with_bootstrap(envs, model, rollout_steps, init_obs):
    obs = init_obs
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []

    for _ in range(rollout_steps):
        obs_buf.append(obs)

        with torch.no_grad():
            action, logp, value = model.act(obs)

        val_buf.append(value)
        logp_buf.append(logp)
        act_buf.append(action)

        actions = action.cpu().numpy()
        next_obs, reward, done, infos = envs.step(actions)

        reward = [get_reward(r) for r in reward]
        # done = np.logical_or(terminated)

        rew_buf.append(torch.tensor(reward, dtype=torch.float32).to(device))
        done_buf.append(torch.tensor(done, dtype=torch.float32).to(device))

        # AsyncVectorEnv resettet automatisch! Kein manueller Reset nötig
        # Wenn done=True, enthält next_obs bereits die neue Episode
        for i, d in enumerate(done):
            if d:
                print(f"Env {i} done. Resetting. (info: {infos[i]})")
                # next_obs[i] ist bereits der Reset-State von AsyncVectorEnv

        obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        max_stage = max([i["stage"] for i in infos])

    with torch.no_grad():
        _, last_value = model.forward(obs)

    obs_buf = torch.stack(obs_buf)
    act_buf = torch.stack(act_buf)
    rew_buf = torch.stack(rew_buf)
    done_buf = torch.stack(done_buf)
    val_buf = torch.stack(val_buf)
    val_buf = torch.cat([val_buf, last_value.unsqueeze(0)], dim=0)
    logp_buf = torch.stack(logp_buf)

    adv_buf, ret_buf = compute_gae_batch(rew_buf, val_buf, done_buf)
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

    return {
        "obs": obs_buf,  # [T, N, obs_dim]
        "actions": act_buf,
        "logprobs": logp_buf,
        "advantages": adv_buf,
        "returns": ret_buf,
        "max_stage": max_stage,
        "last_obs": obs,
    }


def evaluate_policy(env, model, episodes=5, render=False):
    """
    Function to evaluate the learned policy

    Args:
    env: gym.Env single environment (not vector!)

    model: ActorCritic model

    episodes: number of episodes to evaluate

    render: whether to visualize (if True, display on screen)

    Returns:
    avg_return: average total reward
    """
    model.eval()
    total_returns = []
    actions = []
    stages = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        if render:
            env.render()
        while not done:
            obs_tensor = (
                torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1).item()  # greedy action
                actions.append(action)

            obs, reward, done, info = env.step(action)
            stages.append(info["stage"])
            total_reward += reward

        total_returns.append(total_reward)
        info["action_count"] = Counter(actions)
    model.train()
    return np.mean(total_returns), info, max(stages)


def train_ppo():
    num_env = 8
    # WICHTIG: AsyncVectorEnv für echte Parallelisierung auf Apple Silicon!
    # Nutzt mehrere CPU-Cores gleichzeitig → 3-5x schneller
    # Für Mac mit M1/M2/M3 sehr empfohlen!
    try:
        envs = gym.vector.AsyncVectorEnv([lambda: make_env() for _ in range(num_env)])
        print(f"✅ Nutze AsyncVectorEnv: {num_env} Environments laufen parallel auf mehreren CPU-Cores")
    except:
        # Fallback auf SyncVectorEnv falls AsyncVectorEnv Probleme macht
        envs = gym.vector.SyncVectorEnv([lambda: make_env() for _ in range(num_env)])
        print(f"⚠️  Nutze SyncVectorEnv: {num_env} Environments laufen sequenziell (langsamer)")
    
    obs_dim = envs.single_observation_space.shape[-1]
    act_dim = envs.single_action_space.n
    print(f"{obs_dim=} {act_dim=}")
    model = ActorCritic(obs_dim, act_dim).to(device)
    
    # Lade vortrainiertes Modell falls vorhanden, sonst starte von Null
    checkpoint_file = "mario_1_1_ppo.pt"
    start_update = 0
    
    try:
        # PyTorch 2.6+ benötigt weights_only=False für vollständige Checkpoints
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        
        # Falls alter Checkpoint-Format (nur model state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
            model.load_state_dict(checkpoint)
            print(f"✅ Altes Modell geladen: {checkpoint_file}")
            print(f"ℹ️  Optimizer-State nicht verfügbar (alter Checkpoint)")
        # Neues Checkpoint-Format (vollständig)
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_update = checkpoint.get('update', 0)
            print(f"✅ Vollständiger Checkpoint geladen: {checkpoint_file}")
            print(f"📊 Fortsetzen ab Update {start_update}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Modell geladen: {checkpoint_file}")
            
    except FileNotFoundError:
        print(f"ℹ️  Kein vortrainiertes Modell gefunden. Starte Training von Null.")
    
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4)
    
    # Lade Optimizer-State falls vorhanden (nur bei neuem Checkpoint-Format)
    try:
        checkpoint = torch.load(checkpoint_file)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Optimizer-State geladen (Momentum erhalten)")
    except:
        pass

    rollout_steps = 128
    epochs = 4
    minibatch_size = 64
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    eval_env = make_env()
    eval_env.reset()
    
    # Initialisiere CSV-Logger
    csv_filename = init_csv_logger("training_log.csv")

    init_obs = envs.reset()
    update = start_update  # Starte bei gespeichertem Update-Zähler
    while True:
        update += 1
        batch = rollout_with_bootstrap(envs, model, rollout_steps, init_obs)
        init_obs = batch["last_obs"]

        T, N = rollout_steps, envs.num_envs
        total_size = T * N

        obs = batch["obs"].reshape(total_size, *envs.single_observation_space.shape)
        act = batch["actions"].reshape(total_size)
        logp_old = batch["logprobs"].reshape(total_size)
        adv = batch["advantages"].reshape(total_size)
        ret = batch["returns"].reshape(total_size)

        for _ in range(epochs):
            idx = torch.randperm(total_size)
            for start in range(0, total_size, minibatch_size):
                i = idx[start : start + minibatch_size]
                logits, value = model(obs[i])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[i])
                ratio = torch.exp(logp - logp_old[i])
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[i]
                policy_loss = -torch.min(ratio * adv[i], clipped).mean()
                value_loss = (ret[i] - value).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # logging
        avg_return = batch["returns"].mean().item()
        max_stage = batch["max_stage"]
        print(f"Update {update}: avg return = {avg_return:.2f} {max_stage=}")
        
        # Evaluiere bei jedem Update
        avg_score, info, eval_max_stage = evaluate_policy(
            eval_env, model, episodes=1, render=False
        )
        
        # Logge alle Daten in CSV (jedes Update)
        eval_data = {
            'avg_score': avg_score,
            'info': info,
            'eval_max_stage': eval_max_stage
        }
        log_to_csv(csv_filename, update, avg_return, max_stage, eval_data)
        
        # Zeige Evaluations-Ergebnisse nur bei jedem 10. Update an
        if update % 10 == 0:
            print(f"[Eval] Update {update}: avg return = {avg_score:.2f} info: {info}")
            
            if eval_max_stage > 1:
                # Erfolg! Speichere finales Modell
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'update': update,
                    'avg_score': avg_score,
                    'success': True
                }, "mario_1_1_clear.pt")
                print(f"🎉 Level geschafft! Finales Modell gespeichert: mario_1_1_clear.pt")
                break
        if update > 0 and update % 50 == 0:
            # Checkpoint: Speichere ALLES für späteres Fortsetzen
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'avg_return': avg_return,
                'max_stage': max_stage
            }, "mario_1_1_ppo.pt")
            print(f"💾 Checkpoint gespeichert bei Update {update}")


if __name__ == "__main__":
    train_ppo()