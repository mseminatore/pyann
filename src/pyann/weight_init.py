"""Weight initialization strategies."""

from enum import IntEnum
from typing import Union


class WeightInit(IntEnum):
    """Weight initialization types matching libann's Weight_init_type enum."""

    UNIFORM = 0   # WEIGHT_INIT_UNIFORM - uniform in [-limit, limit]
    XAVIER = 1    # WEIGHT_INIT_XAVIER - Xavier/Glorot, suits sigmoid/tanh
    HE = 2        # WEIGHT_INIT_HE - He, suits ReLU variants
    AUTO = 3      # WEIGHT_INIT_AUTO - chosen per layer from its activation (default)

    # Aliases
    GLOROT = 1

    @classmethod
    def from_string(cls, name: str) -> "WeightInit":
        """Convert string name to WeightInit enum.

        Args:
            name: Initializer name (case-insensitive)

        Returns:
            Corresponding WeightInit enum value

        Raises:
            ValueError: If name is not a valid initializer
        """
        name_map = {
            "uniform": cls.UNIFORM,
            "xavier": cls.XAVIER,
            "glorot": cls.XAVIER,
            "he": cls.HE,
            "auto": cls.AUTO,
        }
        key = name.lower().replace("-", "_")
        if key not in name_map:
            valid = ", ".join(sorted(name_map.keys()))
            raise ValueError(f"Unknown weight init '{name}'. Valid options: {valid}")
        return name_map[key]


def parse_weight_init(init: Union[str, WeightInit]) -> WeightInit:
    """Parse weight init argument to WeightInit enum.

    Args:
        init: String name or WeightInit enum

    Returns:
        WeightInit enum value
    """
    if isinstance(init, WeightInit):
        return init
    if isinstance(init, str):
        return WeightInit.from_string(init)
    raise TypeError(f"weight_init must be str or WeightInit, got {type(init)}")
