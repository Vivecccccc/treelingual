import math
import inspect
import time
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import TrainerCallback, Seq2SeqTrainingArguments, GenerationConfig, TrainerState, TrainerControl

from llmtuner.train.ppo.trainer import CustomPPOTrainer
from llmtuner.hparams import ModelArguments, FinetuningArguments, GeneratingArguments
from llmtuner.extras.callbacks import LogCallback, FixValueHeadModelCallback
from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import AverageMeter, count_parameters, get_logits_processor
from llmtuner.train.ppo.utils import dump_layernorm, restore_layernorm

from trainer.tree_rewards import TreeRewards

logger = get_logger(__name__)

class TreePPOTrainer(CustomPPOTrainer, PPOTrainer):
    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: List["TrainerCallback"],
        kb_file_path: str,
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
        tree_rewards = TreeRewards(self.tokenizer, kb_file_path)
        self.reward_function = tree_rewards.batch_award

        self.state = TrainerState()
        self.control = TrainerControl()
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        assert isinstance(self.log_callback, LogCallback) and isinstance(self.save_callback, FixValueHeadModelCallback)
        if self.args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # label => sentiment | we need query and response for logging purpose
            self._signature_columns += ["label", "query", "response"]
            self._signature_columns += ["labels"]

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
            queries, responses, rewards_per_token = [], [], []
            
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch_queries, mini_batch_responses, mini_batch_labels = self.get_inputs(batch[idx:idx+self.config.mini_batch_size])
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_labels, mini_batch_responses)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards_per_token.extend(mini_batch_rewards)

            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False
            self.model.train()

            stats = self.step(queries, responses, rewards_per_token)

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""
        Generates model's responses given queries.
        """
        if self.model_args.upcast_layernorm:
            layernorm_params = dump_layernorm(self.model)

        if batch["input_ids"].size(0) == 1: # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        generate_output: torch.Tensor = unwrapped_model.generate(
            generation_config=self.generation_config,
            logits_processor=get_logits_processor(),
            **batch
        )

        if self.model_args.upcast_layernorm:
            restore_layernorm(self.model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1):].detach().cpu()
        label = batch["labels"].detach().cpu()
        queries, responses, labels = [], [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_index = (response[i] != self.tokenizer.pad_token_id).nonzero()
            label_index = (label[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_index) == 0:
                response_length = 1 # allow empty response
            else:
                response_length = response_index[-1].item() + 1
            label_length = label_index[-1].item() + 1

            queries.append(query[i, query_start_index:]) # remove padding from left
            responses.append(response[i, :response_length]) # remove padding from right
            labels.append(label[i, :label_length]) # remove padding from right
        return queries, responses, labels
    
    def get_rewards(
            self,
            queries: List[torch.Tensor],
            labels: List[torch.Tensor],
            responses: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        rewards, invalid_ids = self.reward_function(labels, responses)
        for i in invalid_ids:
            queries.pop(i)
            labels.pop(i)
            responses.pop(i)
        return rewards

    def step(
            self,
            queries: List[torch.LongTensor],
            responses: List[torch.LongTensor],
            rewards_per_token: List[torch.LongTensor],
            response_masks: Optional[List[torch.LongTensor]] = None
    ):
        bs = len(queries)

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        ) # this method would squeeze scores if dim() == 1, but in our case it won't matter
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling or self.config.score_clip is not None:
            raise NotImplementedError("Score scaling and clipping are not yet implemented")
        
        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            raise NotImplementedError("Distributed training is not yet implemented")
        
        
        