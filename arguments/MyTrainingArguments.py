from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class MyTrainingArguments(TrainingArguments):
    pass
