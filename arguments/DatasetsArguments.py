from dataclasses import dataclass, field


@dataclass
class DatasetsArguments:
    train_csv_path: str = field(default=None)
    test_csv_path: str = field(default=None)
