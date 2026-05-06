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
            epsilon = 0.1,
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
            buffer_capacity = 10_000,
            min_buffer_size = 500,
        )
    )

    training_config = ml_collections.ConfigDict(
        dict(
            learning_rate = 1e-4,
            batch_size = 32,
            gamma = 0.99,
            update_every = 4,
            target_sync_freq = 100,
            n_episodes = 5_000,
            max_t_per_episode = 500,
            check_reward_every = 50,
        )
    )

    config = ml_collections.ConfigDict()
    config.data = data_config
    config.policy = policy_config
    config.log = log_config
    config.buffer = buffer_config
    config.training = training_config
    return ml_collections.FrozenConfigDict(config)
