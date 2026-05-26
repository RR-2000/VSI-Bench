#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastparquet",
#     "pandas",
#     "pathlib",
#     "pyarrow",
# ]
# ///
"""
Create parquet files for config subsets of the VSI-Bench dataset.
* debiased: all examples not pruned by Iterative Bias Pruning (aka VSI-Bench-Debiased)
* pruned: all examples pruned by Iterative Bias Pruning

> [!NOTE]
> If you do not pass `index=False`, the parquet files will have a `__index_level_0__` column
"""

import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent
pruned_ids_path = script_dir / "pruned_ids.txt"
test_jsonl_path = script_dir / "test.jsonl"

bbox_ordering_jsonl_path = script_dir / "bbox_obj_appearance_order.jsonl"
bbox_object_counting_jsonl_path = script_dir / "bbox_object_counting.jsonl"
bbox_object_size_estimation_jsonl_path = script_dir / "bbox_object_size_estimation.jsonl"

pq_debiased_path = script_dir / "test_debiased.parquet"
pq_pruned_path = script_dir / "test_pruned.parquet"
pq_object_counting_path = script_dir / "test_object_counting.parquet"
pq_room_size_estimation_path = script_dir / "test_room_size_estimation.parquet"
pq_object_abs_distance_path = script_dir / "test_object_abs_distance.parquet"
pq_object_size_estimation_path = script_dir / "test_object_size_estimation.parquet"
pq_obj_appearance_order_path = script_dir / "test_object_appearance_order.parquet"
pq_object_rel_direction_hard_path = script_dir / "test_object_rel_direction_hard.parquet"
pq_object_rel_direction_medium_path = script_dir / "test_object_rel_direction_medium.parquet"
pq_object_rel_direction_easy_path = script_dir / "test_object_rel_direction_easy.parquet"

pq_bbox_object_appearance_order_path = script_dir / "bbox_object_appearance_order.parquet"
pq_bbox_object_counting_path = script_dir / "bbox_object_counting.parquet"
pq_bbox_object_size_estimation_path = script_dir / "bbox_object_size_estimation.parquet"

print("Creating parquet files...")

print(f"Loading pruned ids from '{pruned_ids_path}'...")
with open(pruned_ids_path, "r") as f:
    pruned_ids = f.read().splitlines()
print(f"    -> Loaded {len(pruned_ids)} pruned ids.")

print(f"Loading test data from '{test_jsonl_path}'...")
df = pd.read_json(str(test_jsonl_path), lines=True)
print(f"    -> Loaded {len(df)} examples.")
df["pruned"] = df["id"].astype(str).isin(pruned_ids)
print(f"    -> Added pruned column.")

# save the debiased and pruned subsets separately to parquet files
df_debiased = df[~df["pruned"]]
df_pruned = df[df["pruned"]]

# Tasks specific subsets
df_object_counting = df[df["question_type"] == "object_counting"]
df_room_size_estimation = df[df["question_type"] == "room_size_estimation"]
df_object_abs_distance = df[df["question_type"] == "object_abs_distance"]
df_object_size_estimation = df[df["question_type"] == "object_size_estimation"]
df_obj_appearance_order = df[df["question_type"] == "obj_appearance_order"]
df_object_rel_direction_hard = df[df["question_type"] == "object_rel_direction_hard"]
df_object_rel_direction_medium = df[df["question_type"] == "object_rel_direction_medium"]
df_object_rel_direction_easy = df[df["question_type"] == "object_rel_direction_easy"]

# Saveing debiased and pruned subsets to parquet files
print(f"Saving debiased examples to '{pq_debiased_path}'...")
df_debiased.to_parquet(pq_debiased_path, index=False)
print(f"    -> Saved {len(df_debiased)} debiased examples.")

print(f"Saving pruned examples to '{pq_pruned_path}'...")
df_pruned.to_parquet(pq_pruned_path, index=False)
print(f"    -> Saved {len(df_pruned)} pruned examples.")


# Saving task specific subsets to parquet files
print(f"Saving object counting examples to '{pq_object_counting_path}'...")
df_object_counting.to_parquet(pq_object_counting_path, index=False)
print(f"    -> Saved {len(df_object_counting)} object counting examples.")

print(f"Saving room size estimation examples to '{pq_room_size_estimation_path}'...")
df_room_size_estimation.to_parquet(pq_room_size_estimation_path, index=False)
print(f"    -> Saved {len(df_room_size_estimation)} room size estimation examples.")

print(f"Saving object absolute distance estimation examples to '{pq_object_abs_distance_path}'...")
df_object_abs_distance.to_parquet(pq_object_abs_distance_path, index=False)
print(f"    -> Saved {len(df_object_abs_distance)} object absolute distance estimation examples.")

print(f"Saving object size estimation examples to '{pq_object_size_estimation_path}'...")
df_object_size_estimation.to_parquet(pq_object_size_estimation_path, index=False)
print(f"    -> Saved {len(df_object_size_estimation)} object size estimation examples.")

print(f"Saving object appearance order examples to '{pq_obj_appearance_order_path}'...")
df_obj_appearance_order.to_parquet(pq_obj_appearance_order_path, index=False)
print(f"    -> Saved {len(df_obj_appearance_order)} object appearance order examples.")

print(f"Saving object relative direction hard examples to '{pq_object_rel_direction_hard_path}'...")
df_object_rel_direction_hard.to_parquet(pq_object_rel_direction_hard_path, index=False)
print(f"    -> Saved {len(df_object_rel_direction_hard)} object relative direction hard examples.")

print(f"Saving object relative direction medium examples to '{pq_object_rel_direction_medium_path}'...")
df_object_rel_direction_medium.to_parquet(pq_object_rel_direction_medium_path, index=False)
print(f"    -> Saved {len(df_object_rel_direction_medium)} object relative direction medium examples.")

print(f"Saving object relative direction easy examples to '{pq_object_rel_direction_easy_path}'...")
df_object_rel_direction_easy.to_parquet(pq_object_rel_direction_easy_path, index=False)
print(f"    -> Saved {len(df_object_rel_direction_easy)} object relative direction easy examples.")


# Custom clip for bbox object appearance order examples
if bbox_ordering_jsonl_path.exists():
    print(f"Loading bbox object appearance order data from '{bbox_ordering_jsonl_path}'...")
    df_bbox_ordering = pd.read_json(str(bbox_ordering_jsonl_path), lines=True)
    print(f"    -> Loaded {len(df_bbox_ordering)} examples.")
    print(f"Saving bbox object appearance order examples to '{pq_bbox_object_appearance_order_path}'...")
    df_bbox_ordering.to_parquet(pq_bbox_object_appearance_order_path, index=False)
    print(f"    -> Saved {len(df_bbox_ordering)} bbox object appearance order examples.")

# Custom clip for bbox object counting examples
if bbox_object_counting_jsonl_path.exists():
    print(f"Loading bbox object counting data from '{bbox_object_counting_jsonl_path}'...")
    df_bbox_object_counting = pd.read_json(str(bbox_object_counting_jsonl_path), lines=True)
    print(f"    -> Loaded {len(df_bbox_object_counting)} examples.")
    print(f"Saving bbox object counting examples to '{pq_bbox_object_counting_path}'...")
    df_bbox_object_counting.to_parquet(pq_bbox_object_counting_path, index=False)
    print(f"    -> Saved {len(df_bbox_object_counting)} bbox object counting examples.")


# Custom clip for bbox object size estimation examples
if bbox_object_size_estimation_jsonl_path.exists():
    print(f"Loading bbox object size estimation data from '{bbox_object_size_estimation_jsonl_path}'...")
    df_bbox_object_size_estimation = pd.read_json(str(bbox_object_size_estimation_jsonl_path), lines=True)
    print(f"    -> Loaded {len(df_bbox_object_size_estimation)} examples.")
    print(f"Saving bbox object size estimation examples to '{pq_bbox_object_size_estimation_path}'...")
    df_bbox_object_size_estimation.to_parquet(pq_bbox_object_size_estimation_path, index=False)
    print(f"    -> Saved {len(df_bbox_object_size_estimation)} bbox object size estimation examples.")

print("Done.") 
