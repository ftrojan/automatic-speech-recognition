import logging
import datasets


logging.basicConfig(level="INFO", format="[%(levelname)s] %(asctime)s.%(msecs)03d %(name)s.%(funcName)s#%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)
logger.info("started")
minds = datasets.load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
minds = minds.train_test_split(test_size=0.2)
minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
print(minds["train"][0])
logger.info("completed")