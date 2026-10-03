import argparse
from .common import ROOT

def main():
    parser = argparse.ArgumentParser(description="Local Laya accuracy and business-value benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Download and audit pinned public data")
    prepare.add_argument("--test-size", type=int, default=1000, help="Per suite; 0 means full cleaned test split")
    prepare.add_argument("--validation-size", type=int, default=400)
    prepare.add_argument("--seed", type=int, default=20260926)
    prepare.add_argument("--output", default="data/prepared/benchmark.json")
    run = sub.add_parser("run", help="Run both checkpoints and baselines locally")
    run.add_argument("--data", default="data/prepared/benchmark.json")
    run.add_argument("--output", default="results/local")
    run.add_argument("--models", nargs="+", choices=["laya", "laya-multilingual"], default=["laya", "laya-multilingual"])
    run.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--offline", action="store_true")
    report = sub.add_parser("report", help="Rebuild HTML and Markdown from measured results")
    report.add_argument("--input", default="results/local/summary.json")
    verify = sub.add_parser("verify", help="Audit all saved decisions against the frozen dataset and summary")
    verify.add_argument("--input", default="results/full/summary.json")
    args = parser.parse_args()
    if args.command == "prepare":
        if min(args.test_size, args.validation_size) < 0:
            parser.error("Sample sizes cannot be negative")
        from .data import prepare
        prepare(args.output, args.test_size, args.validation_size, args.seed)
    elif args.command == "run":
        if min(args.batch_size, args.threads) < 1:
            parser.error("Batch size and thread count must be positive")
        from .runner import run
        run(args)
    elif args.command == "verify":
        from .verify import verify
        verify(args.input)
    else:
        from .report import generate
        generate(args.input)

if __name__ == "__main__":
    main()
