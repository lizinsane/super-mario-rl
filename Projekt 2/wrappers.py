"""
Custom Wrappers für Super Mario Bros Environment
"""
import numpy as np
from gym import Wrapper


class SkipFrame(Wrapper):
    """
    Wiederholt eine Action für skip Frames und summiert die Rewards.
    Reduziert die Anzahl der zu verarbeitenden Frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class CustomReward(Wrapper):
    """
    Angepasste Reward-Funktion für Mario.
    Belohnt Fortschritt, Zeit und Münzen.
    """
    def __init__(self, env, x_reward=0.1, coin_reward=10.0, death_penalty=50.0, flag_reward=500.0, idle_penalty=0.1):
        super().__init__(env)
        self._current_score = 0
        self._current_x = 0
        self._current_coins = 0
        self._enemies_killed = 0
        
        # Reward-Parameter
        self.x_reward = x_reward
        self.coin_reward = coin_reward
        self.death_penalty = death_penalty
        self.flag_reward = flag_reward
        self.idle_penalty = idle_penalty

    def reset(self, **kwargs):
        self._current_score = 0
        self._current_x = 0
        self._current_coins = 0
        self._enemies_killed = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Belohnung für X-Fortschritt / Bestrafung für Stillstand
        x_progress = info['x_pos'] - self._current_x
        if x_progress > 0:
            # Belohnung für Fortschritt
            reward += x_progress * self.x_reward
        else:
            # Strafe für Stillstand/Rückwärtsbewegung (motiviert Vorwärtsdrang)
            reward -= self.idle_penalty
        
        self._current_x = info['x_pos']
        
        # Belohnung für Münzen
        coins_gained = info.get('coins', 0) - self._current_coins
        if coins_gained > 0:
            reward += coins_gained * self.coin_reward
            self._current_coins = info.get('coins', 0)
        
        # Bestrafung für Tod
        if done and info['life'] < 2:
            reward -= self.death_penalty
        
        # Belohnung für Level-Abschluss
        if info.get('flag_get', False):
            reward += self.flag_reward
            
        return obs, reward, done, info


class ProcessFrame(Wrapper):
    """
    Konvertiert LazyFrames zu numpy array und stellt sicher,
    dass die Dimensionen korrekt sind: (num_frames, height, width)
    """
    def __init__(self, env, frame_stack, resize_shape):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.resize_shape = resize_shape
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._process_observation(obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_observation(obs), reward, done, info
    
    def _process_observation(self, obs):
        """Konvertiert Observation zu korrektem Format"""
        obs = np.array(obs)
        
        # Entferne letzte Dimension falls vorhanden (z.B. (84, 84, 1) -> (84, 84))
        if len(obs.shape) == 4 and obs.shape[-1] == 1:
            obs = obs.squeeze(-1)
        
        # Falls Shape (height, width, channels), transponiere zu (channels, height, width)
        if len(obs.shape) == 3 and obs.shape[-1] == self.frame_stack:
            obs = np.transpose(obs, (2, 0, 1))
        
        return obs
