---
license: apache-2.0
task_categories:
- visual-question-answering
language:
- en
tags:
- Video
- Text
size_categories:
- 1K<n<10K
configs:
  - config_name: full
    data_files:
      - split: test
        path: "test*.parquet"
    default: true
  - config_name: debiased
    data_files:
      - split: test
        path: "test_debiased.parquet"
  - config_name: pruned
    data_files:
      - split: test
        path: "test_pruned.parquet"
  - config_name: object_counting
    data_files:
      - split: test
        path: "test_object_counting.parquet"
  - config_name: object_abs_distance
    data_files:
      - split: test
        path: "test_object_abs_distance.parquet"
  - config_name: object_appearance_order
    data_files:
      - split: test
        path: "test_object_appearance_order.parquet"
  - config_name: object_rel_direction_easy
    data_files:
      - split: test
        path: "test_object_rel_direction_easy.parquet"
  - config_name: object_rel_direction_medium
    data_files:
      - split: test
        path: "test_object_rel_direction_medium.parquet"
  - config_name: object_rel_direction_hard
    data_files:
      - split: test
        path: "test_object_rel_direction_hard.parquet"
  - config_name: object_size_estimation
    data_files:
      - split: test
        path: "test_object_size_estimation.parquet"
  - config_name: room_size_estimation
    data_files:
      - split: test
        path: "test_room_size_estimation.parquet"
  - config_name: bbox_object_counting
    data_files:
      - split: bbox
        path: "bbox_object_counting.parquet"
  - config_name: bbox_object_appearance_order
    data_files:
      - split: bbox
        path: "bbox_object_appearance_order.parquet"
  - config_name: bbox_object_size_estimation
    data_files:
      - split: bbox
        path: "bbox_object_size_estimation.parquet"
  - config_name: baseline_bbox_object_counting
    data_files:
      - split: bbox
        path: "baseline_bbox_object_counting.parquet"
  - config_name: baseline_bbox_object_appearance_order
    data_files:
      - split: bbox
        path: "baseline_bbox_object_appearance_order.parquet"
  - config_name: baseline_bbox_object_size_estimation
    data_files:
      - split: bbox
        path: "baseline_bbox_object_size_estimation.parquet"
---

<!-- <div align="center"> -->

| Dataset | arXiv | Website | Code |
| :------ | :---- | :------ | :--- |
| **VSI-Bench** | <a href="https://arxiv.org/abs/2412.14171" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-thinking--in--space-red?logo=arxiv" height="20" /></a> | <a href="https://vision-x-nyu.github.io/thinking-in-space.github.io/" target="_blank"><img alt="Website" src="https://img.shields.io/badge/🌎_Website-thinking--in--space-blue.svg" height="20" /></a> | <a href="https://github.com/vision-x-nyu/thinking-in-space" target="_blank"><img alt="GitHub Code" src="https://img.shields.io/badge/Code-thinking--in--space-white?&logo=github&logoColor=white" /></a> |
| **VSI-Bench-Debiased** | <a href="https://arxiv.org/abs/2511.04655" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-test--set--stress--test-red?logo=arxiv" height="20" /></a> | <a href="https://vision-x-nyu.github.io/test-set-training/" target="_blank"><img alt="Website" src="https://img.shields.io/badge/🌎_Website-test--set--stress--test-blue.svg" height="20" /></a> | <a href="https://github.com/vision-x-nyu/test-set-training" target="_blank"><img alt="GitHub Code" src="https://img.shields.io/badge/Code-test--set--stress--test-white?&logo=github&logoColor=white" /></a> |

<!-- </div> -->

<br>

> [!IMPORTANT]
> ***[Nov. 7, 2025] UPDATE:** This Dataset has been updated to include a "Debiased" subset following the [TsT Pruning Methodology](https://vision-x-nyu.github.io/test-set-training/)*

<br>

# Visual-Spatial Intelligence Benchmark (VSI-Bench & VSI-Bench-Debiased)

This repository contains the visual spatial intelligence benchmark (VSI-Bench), introduced in [Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces](https://arxiv.org/abs/2412.14171), and its debiased counterpart **VSI-Bench-Debiased**, introduced in our follow-up work on systematic benchmark robustification [Benchmark Designers Should "Train on the Test Set" to Expose Exploitable Non-Visual Shortcuts](https://arxiv.org/abs/2511.04655).

## Overview

**VSI-Bench** evaluates visual-spatial intelligence of multimodal models through egocentric video understanding, comprising over 5,000 question-answer pairs from real-world indoor scenes.

**VSI-Bench-Debiased** is a robustified version that reduces non-visual shortcuts using our Test-set Stress-Test (TsT) and Iterative Bias Pruning (IBP) methodology. This version better isolates visual reasoning capabilities by systematically removing samples that can be solved without visual input.

### Description
VSI-Bench quantitatively evaluates the visual-spatial intelligence of MLLMs from egocentric video. VSI-Bench comprises over 5,000 question-answer pairs derived from 288 real videos. These videos are sourced from the validation sets of the public indoor 3D scene reconstruction datasets `ScanNet`, `ScanNet++`, and `ARKitScenes`, and represent diverse environments -- including residential spaces, professional settings (e.g., offices, labs), and industrial spaces (e.g., factories) and multiple geographic regions. By repurposing these existing 3D reconstruction and understanding datasets, VSI-Bench benefits from accurate object-level annotations, which are used in question generation and could support future studies exploring the connection between MLLMs and 3D reconstruction.


#### Fields

The dataset contains the following fields:

| Field Name | Description |
| :--------- | :---------- |
| `id` | Global index of the entry in the dataset |
| `dataset` | Video source: `scannet`, `arkitscenes` or `scannetpp` |
| `scene_name` | Scene (video) name for each question-answer pair |
| `question_type` | The type of task for question |
| `question` | Question asked about the video |
| `options` | Choices for the question (only for multiple choice questions) |
| `ground_truth` | Ground truth answer for the question |
| `pruned` | Boolean indicating if example was removed by Iterative Bias Pruning (IBP) |

### Why VSI-Bench-Debiased?

While the original VSI-Bench was designed to require visual understanding, our follow-up analysis revealed that a portion of questions could be answered using non-visual shortcuts—such as statistical biases in answer distributions or world knowledge priors—without actually processing the visual input.

**VSI-Bench-Debiased** addresses this through systematic robustification:

1. **Test-set Stress-Test (TsT)**: We applied k-fold cross-validation directly on the test set to identify samples with high non-visual solvability, assigning each sample a bias score.
2. **Iterative Bias Pruning (IBP)**: We iteratively removed samples with the highest bias scores, creating a subset that better compels genuine visual reasoning.

**Key improvements in VSI-Bench-Debiased:**
- **Reduced non-visual solvability**: Blind models (text-only, no vision) perform closer to chance
- **Wider vision-blind gap**: Greater performance difference between vision-enabled and vision-disabled models
- **Better isolation of visual reasoning**: Fine-tuning on in-distribution data improves vision-enabled performance much more than blind performance, confirming reduced shortcut reliance

For researchers interested in robust evaluation of visual-spatial intelligence, **we recommend reporting results on both the full and debiased subsets** to provide comprehensive assessment.

## Usage

### Dataset Configurations

This dataset provides three configurations for flexible evaluation:

| Config | Description | Usage |
|--------|-------------|-------|
| `full` (default) | All 5,131 examples with `pruned` column | Load all data, filter as needed |
| `debiased` | 2,363 examples (non-pruned subset) | Evaluate on robustified benchmark |
| `pruned` | 2,768 examples (pruned by IBP) | Analyze removed samples |

#### Loading the Dataset Annotations

##### Load specific configuration

If you want to load just a specific subset, you can use the config name with the `load_dataset` function as follows:

```python
from datasets import load_dataset

# Load full dataset (default)
vsi_bench_full = load_dataset("nyu-visionx/VSI-Bench")
# or use the config name "full"
vsi_bench_full = load_dataset("nyu-visionx/VSI-Bench", "full")

# Load debiased version only
vsi_bench_debiased = load_dataset("nyu-visionx/VSI-Bench", "debiased")

# Load pruned examples only
vsi_bench_pruned = load_dataset("nyu-visionx/VSI-Bench", "pruned")
```

##### Load full dataset and filter using `pruned` column (recommended)

> [!TIP]
> **For LMMS-Eval users:** We have updated the `vsi-bench` task to automatically report scores on both full and debiased subsets. (TODO: LINK).

We recommend loading the "full" set, evaluating on all samples, and then using the `pruned` column to compute scores on both the full and debiased subsets.

```python
from datasets import load_dataset

# Load full dataset with pruned annotations
vsi_bench_full = load_dataset("nyu-visionx/VSI-Bench")

# Evaluate on full set
model_predictions = evaluate_model(vsi_bench_full)

# Score on both the full and debiased subsets
full_acc = compute_accuracy(model_predictions)
debiased_acc = compute_accuracy(model_predictions.filter(lambda x: not x["pruned"]))
```

### Evaluation

> [!TIP]
> ***TODO: link to the LMMS Eval Code***

VSI-Bench evaluates performance using two metrics: for multiple-choice questions, we use `Accuracy`, calculated based on exact matches. For numerical-answer questions, we introduce a new metric, `MRA (Mean Relative Accuracy)`, to assess how closely model predictions align with ground truth values.

We provide an out-of-the-box evaluation of VSI-Bench in our [GitHub repository](https://github.com/vision-x-nyu/thinking-in-space), including the [metrics](https://github.com/vision-x-nyu/thinking-in-space/blob/main/lmms_eval/tasks/vsibench/utils.py#L109C1-L155C36) implementation used in our framework. For further detailes, users can refer to our paper and GitHub repository.

## Files

- `test-*.parquet`: Parquet files containing dataset annotations (questions, answers, metadata).
  * `test_debiased.parquet`: Annotations for the debiased subset (2,363 examples)
  * `test_pruned.parquet`: Annotations for the pruned subset (2,768 examples)
- `*.zip`: Compressed video files for the dataset
  * `arkitscenes.zip`: Videos for the ARKitScenes dataset
  * `scannet.zip`: Videos for the ScanNet dataset
  * `scannetpp.zip`: Videos for the ScanNet++ dataset
- `pruned_ids.txt`: List of example IDs removed by Iterative Bias Pruning
- `create_pq.py`: Convenience script to regenerate parquet files from `test.jsonl` and `pruned_ids.txt`. Can be run with `uv run create_pq.py`.


## Citation

If you use these datasets in your research, please cite the original VSI-Bench paper and our debiasing paper that produced VSI-Bench-Debiased:

```bibtex
@inproceedings{yang2025thinking,
    title={{Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces}},
    author={Yang, Jihan and Yang, Shusheng and Gupta, Anjali and Han, Rilyn and Fei-Fei, Li and Xie, Saining},
    booktitle={CVPR},
    year={2025},
}

@article{brown2025benchmark,
    title={{Benchmark Designers Should "Train on the Test Set" to Expose Exploitable Non-Visual Shortcuts}},
    author={Brown, Ellis and Yang, Jihan and Yang, Shusheng and Fergus, Rob and Xie, Saining},
    year={2025},
    journal={arXiv preprint arXiv:2511.04655},
}
```
