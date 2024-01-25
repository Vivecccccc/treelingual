import math
from typing import Callable, List
from tqdm import tqdm

from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import TrainerCallback, Seq2SeqTrainingArguments, GenerationConfig, TrainerState, TrainerControl

from llmtuner.train.ppo.trainer import CustomPPOTrainer
from llmtuner.hparams import ModelArguments, FinetuningArguments, GeneratingArguments
from llmtuner.extras.callbacks import LogCallback, FixValueHeadModelCallback
from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import AverageMeter, count_parameters, get_logits_processor

logger = get_logger(__name__)

class TreePPOTrainer(CustomPPOTrainer, PPOTrainer):
    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: List["TrainerCallback"],
        reward_function: Callable,
        **kwargs,
    ):
        PPOTrainer.__init__(self, **kwargs)
        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id],
            **generating_args.to_dict()
        )
        self.reward_function = reward_function

        self.state = TrainerState()
        self.control = TrainerControl()
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        assert isinstance(self.log_callback, LogCallback) and isinstance(self.save_callback, FixValueHeadModelCallback)

    def ppo_train(self) -> None:
        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        len_dataloader = len(self.dataloader)
        num_examples = len(self.dataset)
        num_train_epochs = self.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * len_dataloader)
        steps_in_epoch = len_dataloader
        
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info("  Num examples = {}".format(num_examples))
            logger.info("  Num Epochs = {}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {}".format(self.args.per_device_train_batch_size))
            logger.info("  Total train batch size (w. parallel, buffer, distributed & accumulation) = {}".format(
                total_train_batch_size
            ))
            logger.info("  Gradient Accumulation steps = {}".format(self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {}".format(self.finetuning_args.ppo_epochs))
            logger.info("  Total training steps = {}".format(max_steps))
            logger.info("  Number of trainable parameters = {}".format(count_parameters(self.model)[0]))
        
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)
            
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            self.model.eval()

            self.tokenizer.padding_side = "right"
            queries, responses, rewards = [], [], []
            