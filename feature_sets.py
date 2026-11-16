"""Feature sets for comparative fraud detection models.

Defines the ladder of feature sets:
  - BASE: tabular transaction features and balance error accounting
  - STRUCT: topological graph features (degree, PageRank, triangles, components)
  - COMM: community structure features (LPA sizes and same_community indicator)
  - BEHAV: label-free behavioural aggregation features

Explicitly excludes 'step' and 'isFlaggedFraud' from all model feature lists.
"""

from __future__ import annotations

from typing import Dict, List

# Tabular base features
BASE_NUMERIC: List[str] = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "is_merchant",
]

BASE_CATEGORICAL: List[str] = ["type"]

BASE: List[str] = BASE_NUMERIC + BASE_CATEGORICAL

# Tabular baseline ablation without balance error accounting
BASE_NOERR_NUMERIC: List[str] = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "is_merchant",
]

BASE_NOERR: List[str] = BASE_NOERR_NUMERIC + BASE_CATEGORICAL

# Structural graph features (origin and destination)
STRUCTURAL_FEATURES: List[str] = [
    "origin_in_degree",
    "origin_out_degree",
    "origin_pagerank",
    "origin_triangle_count",
    "origin_component_size",
    "dest_in_degree",
    "dest_out_degree",
    "dest_pagerank",
    "dest_triangle_count",
    "dest_component_size",
]

# Community features (LPA sizes and co-membership)
COMMUNITY_FEATURES: List[str] = [
    "origin_community_size",
    "dest_community_size",
    "same_community",
]

# Behaviour features (label-free aggregations)
BEHAVIOUR_FEATURES: List[str] = [
    "dest_total_in_amount",
    "dest_total_out_amount",
    "dest_pass_through_ratio",
    "dest_distinct_senders",
    "dest_distinct_receivers",
    "origin_total_out_amount",
    "origin_distinct_receivers",
]

# Model feature dictionary
MODEL_FEATURE_SETS: Dict[str, List[str]] = {
    "m0_rule": [],
    "m1_tabular": list(BASE),
    "m1_tabular_noerr": list(BASE_NOERR),
    "m2_graph": list(BASE + STRUCTURAL_FEATURES),
    "m3_community_mule": list(
        BASE + STRUCTURAL_FEATURES + COMMUNITY_FEATURES + BEHAVIOUR_FEATURES
    ),
}

ALL_MODEL_NAMES: List[str] = [
    "m0_rule",
    "m1_tabular",
    "m1_tabular_noerr",
    "m2_graph",
    "m3_community_mule",
]

TRAINABLE_MODELS: List[str] = [
    "m1_tabular",
    "m1_tabular_noerr",
    "m2_graph",
    "m3_community_mule",
]


def get_model_features(model_name: str) -> List[str]:
    """Return the complete feature list for a named model."""
    if model_name not in MODEL_FEATURE_SETS:
        raise KeyError(f"Unknown model: {model_name}. Available: {list(MODEL_FEATURE_SETS)}")
    return list(MODEL_FEATURE_SETS[model_name])


def get_numeric_features(model_name: str) -> List[str]:
    """Return the numeric feature list for a named model."""
    features = get_model_features(model_name)
    return [f for f in features if f not in BASE_CATEGORICAL]


def get_categorical_features(model_name: str) -> List[str]:
    """Return the categorical feature list for a named model."""
    features = get_model_features(model_name)
    return [f for f in features if f in BASE_CATEGORICAL]
