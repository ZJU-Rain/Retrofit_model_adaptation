"""
Centralised configuration entry-point for the our merged pipeline
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AllArgs:
    # ---------------- paths ----------------
    base_model_path: str = (
        "/base_model_path"
    )
    adapter_paths: List[str] = field(default_factory=lambda: [
        "/adapter_path"
    ])
    val_files: List[str] = field(default_factory=lambda: [
        "/val_file/valid_data.jsonl"
    ])
    out_adapter_dir: str = "./output_path"
    out_full_pt_path: str = "./output_path/merged_state_dict.pt"

    # ---------------- data ----------------
    batch_size: int = 4

    # ---------------- retrofit ----------------
    loc_epochs: int = 20
    loc_lr: float = 1e6     # last_task_lr = 1e8
    loc_l1: float = 2.0     # last_task_l1 = 1.0
    loc_l2: float = 0.0     # last_task_l1 = 0.1
    loc_sigmoid_bias: float = 5.0
    loc_sparsity: float = 1.0
    loc_max_batches: int = 68424   # valid data max batches
    loc_sample_cap: Optional[int] = None
    use_kg_loss: bool = False   # last_task_use_kg_loss = True

_DEFAULT_ARGS = AllArgs()

def parse_args() -> AllArgs:
    parser = argparse.ArgumentParser(description="LoRA-B retrofit → merge → export")

    # ---------------- paths ----------------
    parser.add_argument("--base_model_path", type=str,
                        default=_DEFAULT_ARGS.base_model_path)
    parser.add_argument("--adapter_paths", type=str, nargs="+",
                        default=_DEFAULT_ARGS.adapter_paths)
    parser.add_argument("--val_files", type=str, nargs="+",
                        default=_DEFAULT_ARGS.val_files)
    parser.add_argument("--out_adapter_dir", type=str,
                        default=_DEFAULT_ARGS.out_adapter_dir)
    parser.add_argument("--out_full_pt_path", type=str,
                        default=_DEFAULT_ARGS.out_full_pt_path)

    # ---------------- data ----------------
    parser.add_argument("--batch_size", type=int,
                        default=_DEFAULT_ARGS.batch_size)

    # ---------------- retrofit ----------------
    parser.add_argument("--loc_epochs", type=int,
                        default=_DEFAULT_ARGS.loc_epochs)
    parser.add_argument("--loc_lr", type=float,
                        default=_DEFAULT_ARGS.loc_lr)
    parser.add_argument("--loc_l1", type=float,
                        default=_DEFAULT_ARGS.loc_l1)
    parser.add_argument("--loc_l2", type=float,
                        default=_DEFAULT_ARGS.loc_l2)
    parser.add_argument("--loc_sigmoid_bias", type=float,
                        default=_DEFAULT_ARGS.loc_sigmoid_bias)
    parser.add_argument("--loc_sparsity", type=float,
                        default=_DEFAULT_ARGS.loc_sparsity)
    parser.add_argument("--loc_max_batches", type=int,
                        default=_DEFAULT_ARGS.loc_max_batches)
    parser.add_argument("--loc_sample_cap", type=int,
                        default=_DEFAULT_ARGS.loc_sample_cap)
    parser.add_argument("--use_kg_loss", action="store_true",
                        default=_DEFAULT_ARGS.use_kg_loss,
                        help="set this flag to enable KG loss")
    namespace = parser.parse_args()
    return AllArgs(**vars(namespace))