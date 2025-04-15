class IDQNConfig:
    """Configuration class for QMix model.

    IDQNConfig contains parameters used to instantiate a QMix model.
    These parameters define the model architecture, behavior, and training settings.

    Args:
        rnn_hidden_dim (int, optional): Dimension of GRU's hidden state.
        gamma (float, optional): Discount factor in reinforcement learning.
        egreedy_exploration (float, optional): Initial 'epsilon' in epsilon-greedy exploration.
        min_exploration (float, optional): Minimum 'epsilon' in epsilon-greedy.
        target_update_interval (int, optional): Sync parameters to target model after 'target_update_interval' times.
        learning_rate (float, optional): Learning rate of the optimizer.
        min_learning_rate (float, optional): Minimum learning rate of the optimizer.
        clip_grad_norm (float, optional): Clipped value of the global norm of gradients.
        hypernet_layers (int, optional): Number of layers in hypernetwork.
        hypernet_embed_dim (int, optional): Embedding dimension for hypernetwork.
        learner_update_freq (int, optional): Update learner frequency.
        double_q (bool, optional): Use Double-DQN.
        algo_name (str, optional): Name of the algorithm.
    """

    def __init__(
        self,
        rnn_hidden_dim: int = 64,
        gamma: float = 0.99,
        egreedy_exploration: float = 1.0,
        min_exploration: float = 0.1,
        target_update_interval: int = 1000,
        learning_rate: float = 0.0005,
        min_learning_rate: float = 0.0001,
        clip_grad_norm: float = 10,
        learner_update_freq: int = 2,
        double_q: bool = True,
        algo_name: str = "idqn",
    ) -> None:

        # Network architecture parameters
        self.rnn_hidden_dim = rnn_hidden_dim

        # Training parameters
        self.gamma = gamma
        self.egreedy_exploration = egreedy_exploration
        self.min_exploration = min_exploration
        self.target_update_interval = target_update_interval
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.learner_update_freq = learner_update_freq
        self.double_q = double_q

        # Logging parameters
        self.algo_name = algo_name
