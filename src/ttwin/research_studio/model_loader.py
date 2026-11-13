"""Model Checkpoint Loader for the Research Studio.

Provides safe loading, metadata extraction, and schema verification for trained
model checkpoints. Complies strictly with the scientific integrity rule:
never fabricate a model or simulate predictions if no checkpoint exists.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import joblib

from ttwin.research_studio.data_registry import DataRegistry, ModelCheckpointDescriptor


class ModelLoader:
    """Discovers, validates, and loads actual trained model checkpoints."""

    def __init__(self, registry: Optional[DataRegistry] = None):
        self.registry = registry or DataRegistry()

    def list_available_models(self) -> List[ModelCheckpointDescriptor]:
        """Return descriptors of all verified trained models present on disk."""
        return self.registry.list_available_checkpoints()

    def has_checkpoints(self) -> bool:
        """Check whether any compatible model checkpoint is available on disk."""
        return len(self.list_available_models()) > 0

    def load_checkpoint(self, checkpoint_name: Optional[str] = None) -> Tuple[Optional[Any], Optional[ModelCheckpointDescriptor], str]:
        """Load a trained model checkpoint from disk.

        Returns:
            (model_obj, descriptor, status_message)
            If no model exists, returns (None, None, message).
        """
        available = self.list_available_models()
        if not available:
            msg = (
                "Trained checkpoint unavailable. Run the verified training pipeline "
                "to generate a compatible model checkpoint."
            )
            return None, None, msg

        # Select requested model or default to first available
        target_desc = None
        if checkpoint_name:
            for desc in available:
                if desc.name == checkpoint_name:
                    target_desc = desc
                    break
            if target_desc is None:
                return None, None, f"Checkpoint '{checkpoint_name}' not found on disk."
        else:
            target_desc = available[0]

        try:
            if target_desc.format == "txt":
                import lightgbm as lgb
                model_obj = lgb.Booster(model_file=str(target_desc.path))
            else:
                model_obj = joblib.load(target_desc.path)
            return model_obj, target_desc, f"Successfully loaded checkpoint: {target_desc.name}"
        except Exception as e:
            return None, target_desc, f"Failed to load checkpoint from {target_desc.path}: {str(e)}"

    def get_reproducible_training_command(self) -> str:
        """Return the exact command to train and generate compatible baseline models."""
        return "python scripts/train_baseline_model.py"

    def get_v3_experiments_command(self) -> str:
        """Return the command to execute the full v3 scientific experiment suite."""
        return "python scripts/run_v3_experiments.py --force"
