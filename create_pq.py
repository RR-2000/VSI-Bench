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

import glob
import os
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

GEN_MP4 = False  # Set to True if you want to generate mp4 files for the bbox examples (requires ffmpeg and a lot of disk space)

script_dir = Path(__file__).parent
pruned_ids_path = script_dir / "pruned_ids.txt"
test_jsonl_path = script_dir / "test.jsonl"

# BBox Jsonl paths
bbox_ordering_jsonl_path = script_dir / "bbox_obj_appearance_order.jsonl"
bbox_object_counting_jsonl_path = script_dir / "bbox_object_counting.jsonl"
bbox_object_size_estimation_jsonl_path = script_dir / "bbox_object_size_estimation.jsonl"
bbox_object_rel_direction_hard_jsonl_path = script_dir / "bbox_object_rel_direction_hard.jsonl"
bbox_object_abs_distance_jsonl_path = script_dir / "bbox_object_abs_distance.jsonl"
bbox_route_planning_jsonl_path = script_dir / "bbox_route_planning.jsonl"


# Output parquet paths
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

# Output parquet paths for BBoxes PQs
pq_bbox_object_appearance_order_path = script_dir / "bbox_object_appearance_order.parquet"
pq_bbox_object_counting_path = script_dir / "bbox_object_counting.parquet"
pq_bbox_object_size_estimation_path = script_dir / "bbox_object_size_estimation.parquet"
pq_bbox_object_rel_direction_hard_path = script_dir / "bbox_object_rel_direction_hard.parquet"
pq_bbox_object_abs_distance_path = script_dir / "bbox_object_abs_distance.parquet"
pq_bbox_route_planning_path = script_dir / "bbox_route_planning.parquet"

# Output parquet paths for BBoxes PQs compliant with the baseline (i.e. with mp4 paths instead of png folders)
pq_baseline_bbox_object_appearance_order_path = script_dir / "baseline_bbox_object_appearance_order.parquet"
pq_baseline_bbox_object_counting_path = script_dir / "baseline_bbox_object_counting.parquet"
pq_baseline_bbox_object_size_estimation_path = script_dir / "baseline_bbox_object_size_estimation.parquet"
pq_baseline_bbox_object_rel_direction_hard_path = script_dir / "baseline_bbox_object_rel_direction_hard.parquet"
pq_baseline_bbox_object_abs_distance_path = script_dir / "baseline_bbox_object_abs_distance.parquet"
pq_baseline_bbox_route_planning_path = script_dir / "baseline_bbox_route_planning.parquet"

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
    def prune_bbox_ordering(row):
        path_to_test = os.path.join(row["dataset"], row['scene_name'])
        if os.path.exists(f'{path_to_test}.mp4') and os.path.isdir(path_to_test) and len(os.listdir(path_to_test)) == 4:
            return True
        return False
    df_bbox_ordering = df_bbox_ordering[df_bbox_ordering.apply(prune_bbox_ordering, axis=1)]
    print(f"Saving bbox object appearance order examples to '{pq_bbox_object_appearance_order_path}'...")
    df_bbox_ordering.to_parquet(pq_bbox_object_appearance_order_path, index=False)
    print(f"    -> Saved {len(df_bbox_ordering)} bbox object appearance order examples.")

# Custom clip for bbox object counting examples
if bbox_object_counting_jsonl_path.exists():
    print(f"Loading bbox object counting data from '{bbox_object_counting_jsonl_path}'...")
    df_bbox_object_counting = pd.read_json(str(bbox_object_counting_jsonl_path), lines=True)
    print(f"    -> Loaded {len(df_bbox_object_counting)} examples.")
    def prune_bbox_counting(row):
        path_to_test = os.path.join(row["dataset"], row['scene_name'])
        if os.path.exists(f'{path_to_test}.mp4') and os.path.isdir(path_to_test) and len(os.listdir(path_to_test)) == 1:
            return True
        return False
    df_bbox_object_counting = df_bbox_object_counting[df_bbox_object_counting.apply(prune_bbox_counting, axis=1)]
    print(f"Saving bbox object counting examples to '{pq_bbox_object_counting_path}'...")
    df_bbox_object_counting.to_parquet(pq_bbox_object_counting_path, index=False)
    print(f"    -> Saved {len(df_bbox_object_counting)} bbox object counting examples.")


# Custom clip for bbox object size estimation examples
if bbox_object_size_estimation_jsonl_path.exists():
    print(f"Loading bbox object size estimation data from '{bbox_object_size_estimation_jsonl_path}'...")
    df_bbox_object_size_estimation = pd.read_json(str(bbox_object_size_estimation_jsonl_path), lines=True)
    print(f"    -> Loaded {len(df_bbox_object_size_estimation)} examples.")
    def prune_bbox_size_estimation(row):
        path_to_test = os.path.join(row["dataset"], row['scene_name'])
        if os.path.exists(f'{path_to_test}.mp4') and os.path.isdir(path_to_test) and len(os.listdir(path_to_test)) == 1:
            return True
        return False
    df_bbox_object_size_estimation = df_bbox_object_size_estimation[df_bbox_object_size_estimation.apply(prune_bbox_size_estimation, axis=1)]
    print(f"Saving bbox object size estimation examples to '{pq_bbox_object_size_estimation_path}'...")
    df_bbox_object_size_estimation.to_parquet(pq_bbox_object_size_estimation_path, index=False)
    print(f"    -> Saved {len(df_bbox_object_size_estimation)} bbox object size estimation examples.")

# Custom clip for bbox relative direction hard examples
if bbox_object_rel_direction_hard_jsonl_path.exists():
    print(f"Loading bbox object relative direction hard data from '{bbox_object_rel_direction_hard_jsonl_path}'...")
    df_bbox_object_rel_direction_hard = pd.read_json(str(bbox_object_rel_direction_hard_jsonl_path), lines=True)
    print(f"    -> Loaded {len(df_bbox_object_rel_direction_hard)} examples.")
    def prune_bbox_rel_direction_hard(row):
        path_to_test = os.path.join(row["dataset"], row['scene_name'])
        if os.path.exists(f'{path_to_test}.mp4') and os.path.isdir(path_to_test) and len(os.listdir(path_to_test)) > 0:
            return True
        return False
    df_bbox_object_rel_direction_hard = df_bbox_object_rel_direction_hard[df_bbox_object_rel_direction_hard.apply(prune_bbox_rel_direction_hard, axis=1)]
    print(f"Saving bbox object relative direction hard examples to '{pq_bbox_object_rel_direction_hard_path}'...")
    df_bbox_object_rel_direction_hard.to_parquet(pq_bbox_object_rel_direction_hard_path, index=False)
    print(f"    -> Saved {len(df_bbox_object_rel_direction_hard)} bbox object relative direction hard examples.")

# Custom clip for bbox absolute distance examples
if bbox_object_abs_distance_jsonl_path.exists():
    print(f"Loading bbox object absolute distance data from '{bbox_object_abs_distance_jsonl_path}'...")
    df_bbox_object_abs_distance = pd.read_json(str(bbox_object_abs_distance_jsonl_path), lines=True)
    print(f"    -> Loaded {len(df_bbox_object_abs_distance)} examples.")
    def prune_bbox_abs_distance(row):
        path_to_test = os.path.join(row["dataset"], row['scene_name'])
        if os.path.exists(f'{path_to_test}.mp4') and os.path.isdir(path_to_test) and len(os.listdir(path_to_test)) > 0:
            return True
        return False
    df_bbox_object_abs_distance = df_bbox_object_abs_distance[df_bbox_object_abs_distance.apply(prune_bbox_abs_distance, axis=1)]
    print(f"Saving bbox object absolute distance examples to '{pq_bbox_object_abs_distance_path}'...")
    df_bbox_object_abs_distance.to_parquet(pq_bbox_object_abs_distance_path, index=False)
    print(f"    -> Saved {len(df_bbox_object_abs_distance)} bbox object absolute distance examples.")

# Custom clip for bbox route planning examples
if bbox_route_planning_jsonl_path.exists():
    print(f"Loading bbox route planning data from '{bbox_route_planning_jsonl_path}'...")
    df_bbox_route_planning = pd.read_json(str(bbox_route_planning_jsonl_path), lines=True)
    print(f"    -> Loaded {len(df_bbox_route_planning)} examples.")
    def prune_bbox_route_planning(row):
        path_to_test = os.path.join(row["dataset"], row['scene_name'])
        if os.path.exists(f'{path_to_test}.mp4') and os.path.isdir(path_to_test) and len(os.listdir(path_to_test)) > 0:
            return True
        return False
    df_bbox_route_planning = df_bbox_route_planning[df_bbox_route_planning.apply(prune_bbox_route_planning, axis=1)]
    print(f"Saving bbox route planning examples to '{pq_bbox_route_planning_path}'...")
    df_bbox_route_planning.to_parquet(pq_bbox_route_planning_path, index=False)
    print(f"    -> Saved {len(df_bbox_route_planning)} bbox route planning examples.")


suffix = '_frames_6'
datatsets = ['scannet', 'scannetpp', 'arkitscenes']
if GEN_MP4:
    for dataset in datatsets:
        dataset_path = script_dir / f"{dataset}{suffix}"
        saveset_path = script_dir / f"{dataset}{suffix}_mp4"
        saveset_path.mkdir(exist_ok=True)
        for scene in tqdm(os.listdir(dataset_path), desc=f"Processing scenes in {dataset}"):
            scene_path = dataset_path / scene
            if os.path.isdir(scene_path):
                output_path = saveset_path / f"{scene}.mp4"
                if not output_path.exists():
                    jpg_paths = sorted(glob.glob(str(scene_path / "*.png")))
                    if not jpg_paths:
                        continue
                    first_frame = cv2.imread(jpg_paths[0])
                    if first_frame is None:
                        continue
                    height, width = first_frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        12,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
                    print(f"Generating mp4 for {scene}...")
                    for jpg_path in jpg_paths:
                        frame = cv2.imread(jpg_path)
                        if frame is None:
                            continue
                        if frame.shape[1] != width or frame.shape[0] != height:
                            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                        writer.write(frame)
                    writer.release()

# Save BBox compliant frames without BBoxes
# Order
df_bbox_ordering['dataset'] = df_bbox_ordering['dataset'].apply(lambda x: f"{x.split('/')[-1]}{suffix}_mp4")
df_bbox_ordering['scene_name'] = df_bbox_ordering['scene_name'].apply(lambda x: f"{'_'.join(x.split('_')[:-1])}")
print(f"Saving bbox object appearance order examples to '{pq_baseline_bbox_object_appearance_order_path}'...")
df_bbox_ordering.to_parquet(pq_baseline_bbox_object_appearance_order_path, index=False)
print(f"    -> Saved {len(df_bbox_ordering)} bbox object appearance order examples.")

# Counting
df_bbox_object_counting['dataset'] = df_bbox_object_counting['dataset'].apply(lambda x: f"{x.split('/')[-1]}{suffix}_mp4")
df_bbox_object_counting['scene_name'] = df_bbox_object_counting['scene_name'].apply(lambda x: f"{'_'.join(x.split('_')[:-1])}")
print(f"Saving bbox object counting examples to '{pq_baseline_bbox_object_counting_path}'...")
df_bbox_object_counting.to_parquet(pq_baseline_bbox_object_counting_path, index=False)
print(f"    -> Saved {len(df_bbox_object_counting)} bbox object counting examples.")

# Size Estimation
df_bbox_object_size_estimation['dataset'] = df_bbox_object_size_estimation['dataset'].apply(lambda x: f"{x.split('/')[-1]}{suffix}_mp4")
df_bbox_object_size_estimation['scene_name'] = df_bbox_object_size_estimation['scene_name'].apply(lambda x: f"{'_'.join(x.split('_')[:-1])}")
print(f"Saving bbox object size estimation examples to '{pq_baseline_bbox_object_size_estimation_path}'...")
df_bbox_object_size_estimation.to_parquet(pq_baseline_bbox_object_size_estimation_path, index=False)
print(f"    -> Saved {len(df_bbox_object_size_estimation)} bbox object size estimation examples.")

# Relative Direction Hard
df_bbox_object_rel_direction_hard['dataset'] = df_bbox_object_rel_direction_hard['dataset'].apply(lambda x: f"{x.split('/')[-1]}{suffix}_mp4")
df_bbox_object_rel_direction_hard['scene_name'] = df_bbox_object_rel_direction_hard['scene_name'].apply(lambda x: f"{'_'.join(x.split('_')[:-1])}")
print(f"Saving bbox object relative direction hard examples to '{pq_baseline_bbox_object_rel_direction_hard_path}'...")
df_bbox_object_rel_direction_hard.to_parquet(pq_baseline_bbox_object_rel_direction_hard_path, index=False)
print(f"    -> Saved {len(df_bbox_object_rel_direction_hard)} bbox object relative direction hard examples.")

# Absolute Distance
df_bbox_object_abs_distance['dataset'] = df_bbox_object_abs_distance['dataset'].apply(lambda x: f"{x.split('/')[-1]}{suffix}_mp4")
df_bbox_object_abs_distance['scene_name'] = df_bbox_object_abs_distance['scene_name'].apply(lambda x: f"{'_'.join(x.split('_')[:-1])}")
print(f"Saving bbox object absolute distance examples to '{pq_baseline_bbox_object_abs_distance_path}'...")
df_bbox_object_abs_distance.to_parquet(pq_baseline_bbox_object_abs_distance_path, index=False)
print(f"    -> Saved {len(df_bbox_object_abs_distance)} bbox object absolute distance examples.")

# Route Planning
df_bbox_route_planning['dataset'] = df_bbox_route_planning['dataset'].apply(lambda x: f"{x.split('/')[-1]}{suffix}_mp4")
df_bbox_route_planning['scene_name'] = df_bbox_route_planning['scene_name'].apply(lambda x: f"{'_'.join(x.split('_')[:-1])}")
print(f"Saving bbox route planning examples to '{pq_baseline_bbox_route_planning_path}'...")
df_bbox_route_planning.to_parquet(pq_baseline_bbox_route_planning_path, index=False)
print(f"    -> Saved {len(df_bbox_route_planning)} bbox route planning examples.")

print("Done.") 
