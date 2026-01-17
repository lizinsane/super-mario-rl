# Super Mario RL - PPO Project

This repository contains a simple reinforcement learning project for Super Mario Bros using Proximal Policy Optimization (PPO).

## Project Structure

The main project is located in the folder `Projekt 2`.

- **ppo2.py**: Main training script. Trains a PPO agent on Super Mario Bros using parallel environments. The model is trained with the parameters set at the top of the file and saved to `mario_ppo_model.pt`. The best model so far is also saved as `mario_ppo_best.pt` whenever a new best performance is reached.

- **plot_episodes.py**: Plots the training log (episode statistics) for visualizing the agent's progress.

- **eval.py**: Evaluates the latest or best model. By default, it starts at level 1-1. Current training progress: End of Level 1-1. You may need to run the evaluation 5-6 times to see the agent reach the end of the level. Alternatively, a small .mp4 recording is available for demonstration.

## Training Parameters (ppo2.py)

Some important parameters you can adjust in `ppo2.py`:

- `NUM_ENVS`: Number of parallel environments (default: 8)
- `STEPS_PER_UPDATE`: Number of steps per PPO update (default: 256)
- `MAX_UPDATES`: Maximum number of training updates (default: 1000)
- `GAMMA`: Discount factor for future rewards (default: 0.9)
- `GAE_LAMBDA`: Lambda for Generalized Advantage Estimation (default: 0.95)
- `CLIP_EPSILON`: PPO clipping parameter (default: 0.2)
- `LEARNING_RATE`: Learning rate for the optimizer (default: 2.5e-4)
- `PPO_EPOCHS`: Number of epochs per update (default: 4)
- `MINIBATCH_SIZE`: Minibatch size for PPO (default: 64)
- `ENTROPY_COEF_START/END`: Entropy coefficient for exploration (default: 0.04)
- `X_REWARD`, `COIN_REWARD`, `DEATH_PENALTY`, `FLAG_REWARD`, `IDLE_PENALTY`: Reward shaping parameters

## Usage

- **Training:**
  ```
  python ppo2.py
  ```
  This will start training and save checkpoints in the same folder.

- **Plotting:**
  ```
  python plot_episodes.py
  ```
  This will generate plots of the training progress.

- **Evaluation:**
  ```
  python eval.py
  ```
  This will evaluate the latest model. You may need to run it several times to see the agent finish the level.

## Demo Video

test

*Agent playing Super Mario Bros Level 1-1 after training*

---

This project is for educational purposes and demonstrates basic reinforcement learning with PPO on a classic NES game.