from src.dr_alns.Trainer import Trainer, get_parameters

from psp_AlnsEnv import pspAlnsEnv

if __name__ == "__main__":
    # Training the model
    config = get_parameters("pspAlnsEnv.yml")
    env = pspAlnsEnv(config)
    trainer = Trainer(env=pspAlnsEnv, config=config)
    trainer.create_model()
    trainer.train()
