import datetime
import os

import gymnasium as gym
import stable_baselines3
import yaml
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from . import settings as settings


def create_env(env, config, n_workers, env_id, **kwargs):
    """
    Dynamically register and create the environment, ensuring each subprocess registers the environment.
    """

    def make_env():
        def _init():
            # Register the environment for workers
            if env_id not in gym.envs.registry:
                try:
                    register(
                        id=env_id,
                        entry_point=lambda: env(
                            config, **kwargs
                        ),  # Create environment with passed config
                    )
                except Exception as e:
                    print(f"Warning: Unable to register the environment in worker: {e}")

            return gym.make(env_id, **kwargs)

        return _init

    envs = [make_env() for _ in range(n_workers)]

    # Parallelize or use single-threaded environment
    if n_workers > 1:
        vectorized = SubprocVecEnv(envs, start_method="spawn")
    else:
        vectorized = DummyVecEnv(envs)

    # # Add monitoring wrapper
    vectorized = VecMonitor(vectorized)

    return vectorized


def get_parameters(config_file):
    """
    Load config file
    """
    env_params = os.path.join(settings.CONFIG, config_file)
    try:
        with open(env_params, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file not found!")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML: {e}")

    return config


class Trainer:
    """
    Wrapper for stable_baselines3 library.
    """

    def __init__(self, env, config):
        self.config = config
        self.env_id = config["environment"]["env_id"]
        self.model = None
        self.env = env

        self.date = datetime.datetime.now().strftime("%m-%d_%H-%M")

        self._env_path = os.path.join(settings.TRAINED_MODELS, self.env_id)
        self._model_path = None
        self.reloaded = False
        self.done = True
        self.test_state = None
        print(f"Loading path {self._env_path}")

    def create_model(self):
        """
        Creates a new RL Model.
        """
        self._create_model_dir()
        self.n_steps = self.config["main"]["n_steps"]

        # Create environment
        self.training_env = create_env(
            self.env,
            self.config,
            n_workers=self.config["main"]["n_workers"],
            env_id=self.env_id,
        )

        policy_name = self.config["main"]["policy"]
        model_name = self.config["main"]["model"]
        model_params = self.config["models"][model_name]
        print(f"\nCreating {model_name} model...")

        policy = getattr(stable_baselines3.common.policies, policy_name)
        model_object = getattr(stable_baselines3, model_name)

        model_args = dict(
            policy=policy,
            env=self.training_env,
            tensorboard_log=self._model_path,
            **model_params,
        )

        self.model = model_object(**model_args)
        return self

    def _create_model_dir(self):
        """
        Creates a unique subfolder in the environment directory for the current trained model.
        """
        if not os.path.isdir(self._env_path):
            os.makedirs(self._env_path)

        try:
            num = max([int(x.split("_")[0]) for x in os.listdir(self._env_path)]) + 1
        except:
            num = 0

        c = self.config["main"]
        dir_name = (
            f"{c['model']}_{c['policy']}_{c['n_steps']}_{c['n_workers']}_{self.date}"
        )
        self._unique_model_identifier = f"{num}_{dir_name}"
        self._model_path = os.path.join(self._env_path, self._unique_model_identifier)

        # Create the model directory if it doesn't exist
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)

    def _save(self):
        """
        Save the trained model, configuration, and environment script.
        """
        save_dir = self._model_path
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        # Save the model
        model_path = os.path.join(save_dir, "model")
        self.model.save(model_path)
        print(f"Model saved at {model_path}.")

        # Save config file
        config_path = os.path.join(save_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(
                self.config, f, indent=4, sort_keys=False, default_flow_style=False
            )
        print(f"Config saved at {config_path}.")

    def train(self, steps=None):
        """
        Train method.
        """

        try:
            save_every = self.config["main"]["save_every"]
            save_every = (
                save_every / self.config["main"]["n_workers"]
            )  # Adjust for parallel environments
            n_steps = steps if steps is not None else self.n_steps
            self.model.is_tb_set = True

            config = dict(
                total_timesteps=n_steps,
                tb_log_name="tensorboard_logging",
                reset_num_timesteps=True,
            )

            # Print message for stopping training
            print("CTRL + C to stop the training and save.\n")

            # Ensure intermediate models are saved
            checkpoint_dir = os.path.join(self._model_path, "intermediate_models")
            os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

            checkpoint_callback = CheckpointCallback(
                save_freq=save_every,
                save_path=checkpoint_dir,
                name_prefix="intermediate_model",
            )

            callback = checkpoint_callback  # Assigning the callback

            # Start training
            self.model = self.model.learn(callback=callback, **config)

            # Save the final trained model
            self._save()
            print(f"Final model saved in {self._model_path}.")

        except KeyboardInterrupt:
            # Handle manual interruption and save model
            print("Training interrupted. Saving model...")
            self._save()
            print("Final model saved.")

    # def launch_tensorboard(self):
    #     """
    #     Launch TensorBoard for the model's log directory.
    #     """
    #     tensorboard_log_dir = os.path.join('tensorboard_logging')  # TensorBoard logs should be in _model_path
    #     print(f"Launching TensorBoard from {tensorboard_log_dir}...")
    #
    #     # Start TensorBoard process
    #     subprocess.Popen(["tensorboard", "--logdir", tensorboard_log_dir])
    #
    #     # Give TensorBoard a little time to start before we return
    #     time.sleep(3)
    #     print("TensorBoard is running. Open http://localhost:6006/ to view.")
