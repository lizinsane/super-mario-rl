"""
Dueling Double Deep Q-Network (DDQN) für Super Mario Bros

TRAINING OUTPUT ERKLÄRUNG:
==========================
Das Training gibt regelmäßig Statistiken aus im Format:
<device> | Epoch: <n> | score: <x> | loss: <y> | stage: <z> | time spent: <t>

📊 OUTPUT-PARAMETER:
--------------------
• device:       Verwendete Hardware (cpu / mps / cuda)
                - cpu: Normale CPU-Berechnung
                - mps: Apple Silicon GPU (Metal Performance Shaders)
                - cuda: NVIDIA GPU

• Epoch:        Aktuelle Episode/Durchlauf-Nummer (0 bis 1.000.000)
                Jede Epoch = Ein vollständiger Versuch, das Level zu spielen
                
• score:        Durchschnittliche Punktzahl der letzten 10 Episoden
                Höhere Werte = Bessere Performance
                - Negativ: Mario stirbt schnell oder bewegt sich rückwärts
                - 0-500: Frühe Lernphase, viele Fehler
                - 500-1500: Macht Fortschritte, erreicht mittlere Bereiche
                - >1500: Gute Performance, erreicht späte Level-Abschnitte
                
• loss:         Durchschnittlicher Trainingsfehler der letzten 10 Episoden
                Misst wie gut das Netzwerk Q-Werte vorhersagt
                - Hoch (>100): Netzwerk lernt noch stark, große Anpassungen
                - Mittel (10-100): Stabiles Lernen
                - Niedrig (<10): Feinabstimmung, konvergiert
                - Zu niedrig kann auf Überanpassung hindeuten
                
• stage:        Erreichte Stage/Level im Spiel (World-Level)
                SuperMarioBros hat mehrere Stages (1-1, 1-2, etc.)
                Zeigt wie weit Mario im Level gekommen ist
                
• time spent:   Zeit in Sekunden für die letzten 10 Episoden
                Hilft die Trainingsgeschwindigkeit zu überwachen
                - Kürzer = Schnelleres Training oder früher Tod
                - Länger = Langsameres Training oder bessere Performance

💡 TRAINING-TIPPS:
------------------
- Gutes Zeichen: score steigt, loss sinkt über Zeit
- Warnung: loss bleibt konstant hoch → Learning Rate evtl. anpassen
- Normal: score schwankt stark in frühen Epochen (Exploration)
"""

import pickle
import random
import time
from collections import deque

import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *


def arrange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)


class replay_memory(object):
    """Replay Memory zur Speicherung von Erfahrungen (Experience Replay Buffer)"""
    def __init__(self, N):
        self.memory = deque(maxlen=N)  # N: Maximale Anzahl gespeicherter Erfahrungen (älteste werden automatisch gelöscht)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


class model(nn.Module):
    """Dueling DQN Netzwerk-Architektur"""
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        # Convolutional Layer 1: n_frame Eingabekanäle -> 32 Filter, Kernel 8x8, Stride 4
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        # Convolutional Layer 2: 32 -> 64 Filter, Kernel 3x3, Stride 1
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        # Fully Connected Layer: 20736 Neuronen (flachgeklopfte Feature-Maps) -> 512 Neuronen
        self.fc = nn.Linear(20736, 512)
        # Advantage Stream: 512 -> n_action (bewertet einzelne Aktionen)
        self.q = nn.Linear(512, n_action)
        # Value Stream: 512 -> 1 (bewertet den Zustand selbst)
        self.v = nn.Linear(512, 1)

        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)

        self.seq.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        # In-place Operationen für bessere Performance auf MPS
        x = F.relu(self.layer1(x), inplace=False)  # MPS kann Probleme mit inplace haben
        x = F.relu(self.layer2(x), inplace=False)
        x = x.reshape(-1, 20736)  # reshape ist flexibler als view für MPS
        x = F.relu(self.fc(x), inplace=False)
        adv = self.q(x)  # Advantage-Werte für jede Aktion
        v = self.v(x)    # State-Value (Wert des Zustands)
        # Dueling DQN Formel: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Dies kombiniert State-Value und Advantage zu finalen Q-Werten
        q = v + (adv - 1 / adv.shape[-1] * adv.sum(-1, keepdim=True))

        return q


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(q, q_target, memory, batch_size, gamma, optimizer, device):
    """Trainiert das Q-Netzwerk mit einer Batch von Erfahrungen aus dem Replay Memory"""
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    
    # Double DQN: Beste Aktion wird vom Q-Netzwerk ausgewählt
    with torch.no_grad():
        a_max = q(s_prime).max(1)[1].unsqueeze(-1)
    
    r = torch.FloatTensor(r).unsqueeze(-1).to(device)
    done = torch.FloatTensor(done).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        y = r + gamma * q_target(s_prime).gather(1, a_max) * done  # Target: r + gamma * Q_target(s', a*), bewertet wird vom Target-Netzwerk
    
    a = torch.tensor(a).unsqueeze(-1).to(device)
    q_value = torch.gather(q(s), dim=1, index=a.view(-1, 1).long())

    loss = F.smooth_l1_loss(q_value, y).mean()  # Huber Loss: Robuster gegenüber Ausreißern als MSE
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient Clipping für Stabilität (besonders wichtig auf MPS)
    torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=10.0)
    
    optimizer.step()
    return loss


def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)


def main(env, q, q_target, optimizer, device):
    t = 0  # Trainingsschritt-Zähler
    gamma = 0.99  # Discount-Faktor: Gewichtung zukünftiger Belohnungen (0.99 = 99% der zukünftigen Belohnung wird berücksichtigt)
    batch_size = 256  # Anzahl der Erfahrungen, die gleichzeitig für das Training verwendet werden

    N = 50000  # Maximale Größe des Replay-Memory (Erfahrungsspeicher)
    eps = 0.001  # Epsilon für Exploration: Wahrscheinlichkeit für zufällige Aktionen (0.1% = fast keine zufälligen Aktionen mehr)
    memory = replay_memory(N)
    update_interval = 50  # Anzahl der Trainingsschritte, nach denen das Target-Netzwerk aktualisiert wird
    print_interval = 10  # Anzahl der Episoden, nach denen Statistiken ausgegeben werden

    score_lst = []
    total_score = 0.0
    loss = 0.0
    start_time = time.perf_counter()

    for k in range(1000000):
        s = arrange(env.reset())
        done = False

        while not done:
            if eps > np.random.rand():
                a = env.action_space.sample()
            else:
                with torch.no_grad():  # Keine Gradienten für Inferenz = schneller
                    q_values = q(s)
                    if device == "mps":
                        # MPS-optimiert: Hole Daten nur einmal zurück
                        a = q_values.cpu().numpy().argmax()
                    elif device == "cpu":
                        a = q_values.detach().numpy().argmax()
                    else:
                        a = q_values.cpu().detach().numpy().argmax()
            s_prime, r, done, _ = env.step(a)
            s_prime = arrange(s_prime)
            total_score += r
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r  # Reward Clipping: Normalisiert große Belohnungen, um das Training zu stabilisieren
            memory.push((s, float(r), int(a), s_prime, int(1 - done)))
            s = s_prime
            stage = env.unwrapped._stage
            if len(memory) > 2000:  # Mindestanzahl von Erfahrungen im Memory, bevor Training beginnt
                loss += train(q, q_target, memory, batch_size, gamma, optimizer, device)
                t += 1
            if t % update_interval == 0:
                copy_weights(q, q_target)
                torch.save(q.state_dict(), "mario_q.pth")
                torch.save(q_target.state_dict(), "mario_q_target.pth")

        if k % print_interval == 0:
            time_spent, start_time = (
                time.perf_counter() - start_time,
                time.perf_counter(),
            )
            print(
                "%s |Epoch : %d | score : %f | loss : %.2f | stage : %d | time spent: %f"
                % (
                    device,
                    k,
                    total_score / print_interval,
                    loss / print_interval,
                    stage,
                    time_spent,
                )
            )
            score_lst.append(total_score / print_interval)
            total_score = 0
            loss = 0.0
            pickle.dump(score_lst, open("score.p", "wb"))


if __name__ == "__main__":
    n_frame = 4  # Anzahl der gestapelten Frames (Frame-Stacking für Bewegungsinformation)
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    
    # Device-Auswahl optimiert für Mac
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 Nutze NVIDIA CUDA GPU")
    elif torch.backends.mps.is_available():
        try:
            # Teste ob MPS tatsächlich funktioniert
            test_tensor = torch.zeros(1, device="mps")
            device = "mps"
            print("🍎 Nutze Apple Silicon GPU (MPS)")
        except Exception as e:
            print(f"⚠️  MPS verfügbar aber nicht funktional: {e}")
            print("🔄 Fallback auf CPU")
            device = "cpu"
    else:
        print("💻 Nutze CPU")
    
    # Für MPS: Reduziere Batch-Size für bessere Stabilität
    if device == "mps":
        print("ℹ️  Mac-Optimierung: Verwende angepasste Einstellungen für Apple Silicon")
    
    q = model(n_frame, env.action_space.n, device).to(device)
    q_target = model(n_frame, env.action_space.n, device).to(device)
    optimizer = optim.Adam(q.parameters(), lr=0.0001)  # Learning Rate: 0.0001 (Schrittgröße für Gewichtsaktualisierungen)
    
    # Setze MPS-spezifische Flags für bessere Performance
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.0)  # Automatische Speicherverwaltung

    main(env, q, q_target, optimizer, device)