"""ModelStore: the trained models of this sandbox, one pair of files per model.

  <folder>/<name>.joblib   the estimator and its features
  <folder>/<name>.json     the card: the model as an asset (GET /models, GET /models/{name})

joblib files are loaded only from this folder, which only this sandbox writes: loading a joblib file from
someone else can run arbitrary code. Models move between sandboxes as plain numbers instead (src/ml/sharing.py:
GET /models/{name}/export, and the load operator). A model saved again under the same name replaces the old one.
"""

import json
import re
from pathlib import Path
from typing import Any

import joblib

from src.exceptions import ChainError
from src.ml.trained import TrainedModel

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")


def check_name(name: str) -> str:
    """Model names become file names, so only letters, digits and _ . - are allowed."""
    if not _NAME.fullmatch(name) or ".." in name:
        raise ChainError(f"a model name may hold only letters, digits and _ . -, not {name!r}")
    return name


class ModelStore:
    def __init__(self, folder: Path) -> None:
        self._folder = Path(folder)

    def save(self, model: TrainedModel) -> None:
        check_name(model.name)
        self._folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": model.estimator, "spec": model.spec}, self._artifact(model.name))
        self._card_file(model.name).write_text(json.dumps(model.card, indent=2))

    def load(self, name: str) -> TrainedModel:
        card = self.card(name)
        if card is None:
            raise ChainError(f"no model named {name}; the models are {self.names()}")
        if "recipe" not in card:
            raise ChainError(
                f"{name} was made by fit, which this version no longer has; train it again with lag, map and train"
            )
        stored = joblib.load(self._artifact(name))
        return TrainedModel(name, stored["estimator"], stored["spec"], card)

    def update_card(self, name: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Writes `changes` on the card of a model (evaluate writes its scores there) and returns the card."""
        card = self.card(name)
        if card is None:
            raise ChainError(f"no model named {name}; the models are {self.names()}")
        card.update(changes)
        self._card_file(name).write_text(json.dumps(card, indent=2))
        return card

    def card(self, name: str) -> dict[str, Any] | None:
        check_name(name)
        file = self._card_file(name)
        return json.loads(file.read_text()) if file.exists() else None

    def cards(self) -> list[dict[str, Any]]:
        return [json.loads(file.read_text()) for file in sorted(self._folder.glob("*.json"))]

    def names(self) -> list[str]:
        return [file.stem for file in sorted(self._folder.glob("*.json"))]

    def _artifact(self, name: str) -> Path:
        return self._folder / f"{name}.joblib"

    def _card_file(self, name: str) -> Path:
        return self._folder / f"{name}.json"
