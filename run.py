import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import datasets
import transformers
import evaluate


def uppercase(example):
    return {"transcription": example["transcription"].upper()}


def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


@dataclass
class DataCollatorCTCWithPadding:

    processor: transformers.AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


logging.basicConfig(level="INFO", format="[%(levelname)s] %(asctime)s.%(msecs)03d %(name)s.%(funcName)s#%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)
logger.info("started")
# load dataset
minds = datasets.load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
minds = minds.train_test_split(test_size=0.2)
minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
print(minds["train"][0])
# preprocess
processor = transformers.AutoProcessor.from_pretrained("facebook/wav2vec2-base")
minds = minds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
minds = minds.map(uppercase)
encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
print(minds["train"][0])
data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
# evaluate
wer = evaluate.load("wer")  # word error rate [0, 1], the smaller the better
logger.info("completed")