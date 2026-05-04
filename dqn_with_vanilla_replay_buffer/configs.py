import ml_collections

def get_configs():
    data_config = ml_collections.ConfigDict(
        dict(
            obs_shape = (96, 96, 3),
            n_actions = 5,
        )
    )

    model_config = ml_collections.ConfigDict(
        dict(
            dropout_rate = 0.1,
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
            update_every = 10,
            temperature = 1.0,
            target_sync_freq = 1000,
            n_episodes = 2000,
            max_t_per_episode = 1000,
            check_reward_every = 10,
        )
    )

    config = ml_collections.ConfigDict()
    config.data = data_config
    config.model = model_config
    config.log = log_config
    config.buffer = buffer_config
    config.training = training_config
    return ml_collections.FrozenConfigDict(config)
