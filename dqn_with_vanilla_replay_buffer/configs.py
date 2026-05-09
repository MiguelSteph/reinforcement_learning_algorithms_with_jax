import ml_collections


def get_configs():
    data_config = ml_collections.ConfigDict(
        dict(
            obs_shape = (84, 84, 4),
            n_actions = 4,
        )
    )

    policy_config = ml_collections.ConfigDict(
        dict(
            epsilon_start       = 1.0,
            epsilon_end         = 0.05,
            epsilon_decay_steps = 500_000,
        )
    )
            
    log_config = ml_collections.ConfigDict(
        dict(
            log_metric_path = 'log_dir/metrics',
            log_checkpoint_path = 'log_dir/checkpoints',
        )
    )

    buffer_config = ml_collections.ConfigDict(
        dict(
            buffer_capacity = 50000,
        )
    )

    training_config = ml_collections.ConfigDict(
        dict(
            learning_rate = 1e-4,
            batch_size = 64,
            gamma = 0.99,
            update_every = 4,
            target_sync_freq = 1000,
            n_episodes = 20000,
            max_t_per_episode = 2000,
            check_reward_every = 100,
        )
    )

    config = ml_collections.ConfigDict()
    config.data = data_config
    config.policy = policy_config
    config.log = log_config
    config.buffer = buffer_config
    config.training = training_config
    return ml_collections.FrozenConfigDict(config)
