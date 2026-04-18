"""
BansuriAI-V2 — Note Mapper

Maps integer class indices to Indian classical swara labels and back.

The bansuri's natural scale in the key of the flute covers seven swaras:
    0 → Sa  (Shadja)     — the tonic / root
    1 → Re  (Rishabh)    — second degree
    2 → Ga  (Gandhar)    — third degree
    3 → Ma  (Madhyam)    — fourth degree
    4 → Pa  (Pancham)    — fifth degree
    5 → Dha (Dhaivat)    — sixth degree
    6 → Ni  (Nishad)     — seventh degree

This module is the single authority for the label vocabulary.
Every other module that needs note names imports from here.
"""

# Ordered list — index position IS the class index
SWARA_LABELS: list[str] = ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"]

# Reverse lookup: name → index
SWARA_TO_INDEX: dict[str, int] = {
    label: idx for idx, label in enumerate(SWARA_LABELS)
}


def index_to_swara(index: int) -> str:
    """Convert a class index to its swara name.

    Args:
        index: Integer class index (0–6).

    Returns:
        Swara label string (e.g. "Sa", "Re").

    Raises:
        IndexError: If index is out of the valid range.
    """
    if not 0 <= index < len(SWARA_LABELS):
        raise IndexError(
            f"Class index {index} out of range. "
            f"Valid range: 0–{len(SWARA_LABELS) - 1}"
        )
    return SWARA_LABELS[index]


def swara_to_index(name: str) -> int:
    """Convert a swara name to its class index.

    Args:
        name: Swara label string (e.g. "Sa", "Re"). Case-sensitive.

    Returns:
        Integer class index.

    Raises:
        KeyError: If name is not a recognized swara.
    """
    if name not in SWARA_TO_INDEX:
        raise KeyError(
            f"Unknown swara '{name}'. "
            f"Valid swaras: {', '.join(SWARA_LABELS)}"
        )
    return SWARA_TO_INDEX[name]


def get_num_classes() -> int:
    """Return the total number of note classes."""
    return len(SWARA_LABELS)
