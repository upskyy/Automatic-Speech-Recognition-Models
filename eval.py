import torch
import numpy as np
import random
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from evaluator.evaluator import Evaluator
from model_builder import load_test_model
from data.data_loader import (
    SpectrogramDataset,
    AudioDataLoader,
)
from vocabulary import (
    load_label,
    load_dataset,
)

from data import MelSpectrogramConfig
from evaluator import EvaluateConfig


cs = ConfigStore.instance()
cs.store(group="audio", name="melspectrogram", node=MelSpectrogramConfig, package="audio")
cs.store(group="eval", name="default", node=EvaluateConfig, package="eval")


@hydra.main(config_path='configs', config_name='eval')
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    torch.manual_seed(config.eval.seed)
    torch.cuda.manual_seed_all(config.eval.seed)
    np.random.seed(config.eval.seed)
    random.seed(config.eval.seed)

    use_cuda = config.eval.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    char2id, id2char = load_label(config.eval.label_path, config.eval.blank_id)
    audio_paths, transcripts, _, _ = load_dataset(config.eval.dataset_path, config.eval.mode)

    test_dataset = SpectrogramDataset(
        config.eval.audio_path,
        audio_paths,
        transcripts,
        config.audio.sampling_rate,
        config.audio.n_mel,
        config.audio.frame_length,
        config.audio.frame_stride,
        config.audio.extension,
        config.train.sos_id,
        config.train.eos_id,
    )
    test_loader = AudioDataLoader(
        test_dataset,
        batch_size=config.eval.batch_size,
        num_workers=config.eval.num_workers,
    )

    model = load_test_model(config, device)

    print('Start Inference !!!')

    evaluator = Evaluator(config, device, test_loader, id2char)
    evaluator.evaluate(model)


if __name__ == "__main__":
    main()
