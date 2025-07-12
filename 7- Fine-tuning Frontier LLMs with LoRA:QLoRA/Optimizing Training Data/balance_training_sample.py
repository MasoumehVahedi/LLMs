import random
import numpy as np
from typing import Dict





def buildTrainingSample(slots: Dict,
                        dataset_name: str,
                        high_value_cutoff: int = 240,
                        max_per_slot: int = 1200,
                        dataset_name_weight: int = 1,
                        other_weight: int = 5,
                        seed: int = 42
                        ):
    """
        Generate a balanced training sample from price‐binned slots.

        Args:
            slots:            Mapping from rounded price → list of Item objects.
            high_value_cutoff: Prices ≥ this get fully included.
            max_per_slot:     Max number of items to sample from lower‐value slots.
            automotive_weight: Weight for items in the 'Automotive' category.
            other_weight:     Weight for all other categories.
            seed:             Random seed for reproducibility.

        Returns:
            A flat list of sampled Item objects.
    """
    # 1. Seed both numpy and random for reproducible sampling
    np.random.seed(seed)
    random.seed(seed)

    sample = []
    # 2. Iterate through each price slot (e.g., 1–999)
    for price, bucket in slots.items():
        # If price is high-value, include every item
        if price >= high_value_cutoff:
            sample.extend(bucket)
        else:
            # If slot is small enough, just include all of it
            if len(bucket) <= max_per_slot:
                sample.extend(bucket)
            else:
                # Build a weight array: automotive items get lower weight
                weights = np.array([dataset_name_weight if record.category==dataset_name else other_weight for record in bucket],
                                   dtype=float)
                weights = weights / np.sum(weights)    # normalize to sum to 1
                # Sample `max_per_slot` records without replacement
                chosen_idxs = np.random.choice(
                    len(bucket),
                    size=max_per_slot,
                    replace=False,
                    p=weights
                )
                # Add the selected records to our sample
                chosen = [bucket[idx] for idx in chosen_idxs]
                sample.extend(chosen)
    return sample




