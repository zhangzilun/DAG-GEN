
# dag-gen-gnn

A constraint-aware workflow DAG generator with few-shot guidance for real-time scheduling evaluation.

## Overview

This repository contains the implementation of **dag-gen-gnn**, a workflow DAG generation framework for benchmark workload synthesis in real-time scheduling and allocation studies. The method combines:

- few-shot style encoding from a small reference set,
- layered constraint-aware DAG generation,
- budget-aware edge-time assignment under critical-path constraints.

## Repository Structure

- `src/`: core source code
- `configs/`: experiment configuration files
- `data/`: DAG datasets and few-shot support sets
- `checkpoints/`: saved model weights
- `outputs/`: generated DAGs and visualisation results

## Environment

Recommended environment:

- Python 3.10
- PyTorch
- NetworkX
- NumPy
- Matplotlib

Install dependencies:

`pip install -r requirements.txt`

## Data

The project uses `.gpickle` workflow DAG datasets.

- `data/gpickle2/`: original DAG dataset
- `data/gpickle2_ip300/`: DAG dataset with critical-path budget 300
- `data/test10/`: few-shot reference DAGs for testing

## Main Scripts

- `src/infer_fewshot_structure1_ratio_v2_timehead.py`: few-shot DAG generation with time-head calibration
- `src/infer_fewshot_structure1_ratio_v2.py`: structure-only inference
- `src/visualize_taskset.py`: visualise generated DAGs
- `src/test_fewshot.py`: few-shot testing utilities

## Run

Generate DAGs:

`python src/infer_fewshot_structure1_ratio_v2_timehead.py --config configs/exp_lp300_fewshot10_timehead.json`

Visualise generated DAGs:

`python src/visualize_taskset.py`

## Paper

This code accompanies the paper:

**A Constraint-aware DAG Taskset Generator with Few-shot Guidance for Real-Time Scheduling Evaluation**

## Notes

This repository is organised for research reproducibility and paper submission. Some datasets and checkpoints may be reduced for storage reasons.
