"""CLI entrypoint exposing the `setup` and `run` subcommands.

A thin argument-parsing shell: all orchestration lives in the setup pipeline.
There are deliberately no hardware or precision flags -- tiering is
deterministic and internal (docs/architecture.md, Decision 1).
"""

from __future__ import annotations

import argparse
import json
import sys

from indic_runner.setup.decision_matrix import TASKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indic-runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="AOT-compile a model into a manifest")
    setup_parser.add_argument("model", help="registry alias or Hugging Face repo id")
    setup_parser.add_argument(
        "--task",
        choices=TASKS,
        help="required when a model serves several tasks, or is not in the registry",
    )
    setup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the plan and print the manifest without downloading anything",
    )
    setup_parser.add_argument(
        "--force", action="store_true", help="rebuild over an existing setup"
    )

    run_parser = subparsers.add_parser("run", help="Stream a dataset through a compiled model")
    run_parser.add_argument("--model", required=True, help="registry alias set up earlier")
    run_parser.add_argument("--dataset", required=True, help="JSONL dataset")
    run_parser.add_argument("--max-new-tokens", type=int, default=512)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--num-beams", type=int, default=1)
    run_parser.add_argument("--timeout", type=float, default=120.0, help="per-row seconds")
    run_parser.add_argument("--retries", type=int, default=1)
    run_parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    run_parser.add_argument("--ocr-variant", choices=["normal", "scanned"])

    return parser


def _print_plan(result) -> None:
    hw, plan, model = result.hardware, result.plan, result.model
    print(f"target      {result.target.repo_id}  (alias: {result.target.alias})")
    print(f"task        {result.target.task}")
    print(
        f"hardware    {hw.os}/{hw.arch} {hw.accelerator}"
        + (f" [{hw.gpu_name}, {hw.vram_gb}GB VRAM]" if hw.gpu_name else "")
        + f"  budget {hw.memory_budget_gb:.1f}GB"
    )
    print(f"model       {model.param_count_b}B params, ctx {model.context_length}"
          + (", encoder-decoder" if model.is_encoder_decoder else ""))
    print(f"engine      {plan.engine} ({plan.mode})")
    print(f"precision   {plan.precision}  ->  {plan.artifact_format}")
    print(_auth_line())
    print()


def _auth_line() -> str:
    """Report whether a token is configured, never what it is."""
    from indic_runner.config import find_env_file, hf_token

    if hf_token() is None:
        return "hf auth     none (gated models will be refused; see .env.example)"
    source = find_env_file()
    return f"hf auth     token configured{f' via {source}' if source else ''}"


def _cmd_setup(args: argparse.Namespace) -> int:
    from indic_runner.setup.pipeline import run_setup

    result = run_setup(
        args.model, task=args.task, dry_run=args.dry_run, force=args.force
    )
    _print_plan(result)
    if result.dry_run:
        print("dry run - manifest that would be written:")
        print(json.dumps(result.manifest, indent=2))
    else:
        print(f"manifest written to {result.manifest_file}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from indic_runner.runtime.engines.base import GenerationConfig
    from indic_runner.runtime.manifest_loader import load_manifest
    from indic_runner.runtime.runner import RunOptions, run

    manifest = load_manifest(args.model)
    options = RunOptions(
        gen=GenerationConfig(args.max_new_tokens, args.temperature, args.num_beams, args.timeout),
        max_retries=args.retries,
        resume_run_id=args.resume,
        ocr_variant=args.ocr_variant,
    )
    result = run(manifest, Path(args.dataset), options)
    print(f"run {result.run_id}: {result.status} "
          f"({result.successful} ok, {result.failed} failed)")
    print(f"results written to {result.results_file}")
    return 0 if result.status == "completed" else 2


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "setup":
        try:
            sys.exit(_cmd_setup(args))
        except Exception as exc:  # noqa: BLE001 - CLI surface: report, don't trace
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.command == "run":
        try:
            sys.exit(_cmd_run(args))
        except Exception as exc:  # noqa: BLE001 - CLI surface: report, don't trace
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
