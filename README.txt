README for Dueling CNN DQN Implementation
Overview

This script implements a Double Dueling Deep Q-Network (DQN) Agent using convolutional neural networks (CNN) for reinforcement learning tasks. It is designed to work with Gymnasium environments, specifically for playing Atari games. The agent architecture incorporates two streams: one for calculating state values and another for advantages for each action, merged to approximate Q-values.
Requirements

    Python 3.x
    gymnasium (gym)
    PyTorch (torch)
    NumPy (numpy)
    TensorBoard for PyTorch (torch.utils.tensorboard)

Features

    Dueling CNN Architecture: For state value and action advantage estimation.
    Double DQN: Utilizes a target network for stable learning.
    Replay Memory: For experience replay.
    TensorBoard Integration: For training visualization and logging.
    Environment Interaction: Works with Gymnasium's vectorized environments.

Usage

    Setup Environment: Define your Gymnasium environment. The script is tailored for Atari game environments (e.g., ALE/Breakout-v5).

    Initialize Agent: Create an instance of DoubleDuelingDQNAgent with appropriate dimensions for state and action spaces.

    Training: Call train_agent function with the environment and agent. Adjust the training parameters as needed (e.g., learning rate, epsilon decay).

    Saving and Loading Models: Use save_model and load_pretrained_model for checkpointing and resuming training.

    Watching Agent Play: Use watch_agent_play to visually inspect the agent's performance in the environment.

Key Classes and Functions

    DuelingCNNDQN: Defines the neural network architecture.
    DoubleDuelingDQNAgent: Manages agent's actions, memories, and learning.
    train_agent: Orchestrates the training process.
    save_model, load_pretrained_model: For model persistence.
    watch_agent_play: Visualize agent's gameplay.

Logging and Visualization

    The script integrates TensorBoard for monitoring training progress. Use TensorBoard to view training metrics like episode rewards.

Additional Notes

    Ensure CUDA compatibility for GPU acceleration (if available).
    Adjust hyperparameters and network architecture based on specific environment requirements.
    The script includes functions for file management (e.g., save_file) related to saving scripts and models.

Disclaimer

    This script is a template and may require modifications to work with different environments or to incorporate additional features.
