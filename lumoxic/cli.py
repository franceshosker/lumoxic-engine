"""CLI interface for the Lumoxic photon computing engine."""

import argparse
import sys
import json

from lumoxic.client import Client


def cmd_process(args: argparse.Namespace) -> None:
    client = Client(threshold=args.threshold, seed=args.seed)
    result = client.process(
        wavelength=args.wavelength,
        angle=args.angle,
        bounces=args.bounces,
    )
    print(f"Binary output: {result.binary_stream.as_string}")
    print(f"Bits: {result.binary_stream.length}")
    print(f"Entropy: {result.binary_stream.entropy:.4f}")
    print(f"Bounces: {len(result.bounces)}")
    print(f"Energy loss: {result.total_energy_loss:.4f}")
    print(f"Time: {result.processing_time_ms:.2f}ms")


def cmd_simulate(args: argparse.Namespace) -> None:
    client = Client(threshold=args.threshold, seed=args.seed)
    result = client.simulate(
        photon_count=args.photons,
        wavelength=args.wavelength,
        max_bounces=args.bounces,
    )
    analysis = client.analyze(result.binary_stream.bits)
    print(f"Photons: {result.photon_count}")
    print(f"Total bits: {result.binary_stream.length}")
    print(f"Binary: {result.binary_stream.as_string[:64]}{'...' if result.binary_stream.length > 64 else ''}")
    print(f"Entropy: {analysis['entropy']:.4f} bits/symbol")
    print(f"Frequency: 0={analysis['frequency'][0]:.3f} 1={analysis['frequency'][1]:.3f}")
    print(f"Time: {result.processing_time_ms:.2f}ms")

    if args.json:
        out = {
            "photon_count": result.photon_count,
            "binary_length": result.binary_stream.length,
            "binary": result.binary_stream.as_string,
            "entropy": analysis["entropy"],
            "frequency": analysis["frequency"],
            "processing_time_ms": result.processing_time_ms,
        }
        print(json.dumps(out, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    from lumoxic.models.training import Trainer
    from lumoxic.models.lnbe import LNBEModel

    model = LNBEModel()
    trainer = Trainer(model)
    data, targets = trainer.generate_training_data(args.samples)
    history = trainer.train(data, targets, epochs=args.epochs)

    print(f"Training complete: {args.epochs} epochs, {args.samples} samples")
    print(f"Final loss: {history[-1]:.6f}")
    print(f"Model: {model.summary()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lumoxic",
        description="Lumoxic AI — Photon Computing Engine",
    )
    parser.add_argument("--seed", type=int, default=None)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("process", help="Process a single photon")
    p.add_argument("--wavelength", type=float, default=450.0)
    p.add_argument("--angle", type=float, default=37.0)
    p.add_argument("--bounces", type=int, default=8)
    p.add_argument("--threshold", type=float, default=45.0)

    s = sub.add_parser("simulate", help="Run batch photon simulation")
    s.add_argument("--photons", type=int, default=256)
    s.add_argument("--wavelength", type=float, default=450.0)
    s.add_argument("--bounces", type=int, default=8)
    s.add_argument("--threshold", type=float, default=45.0)
    s.add_argument("--json", action="store_true")

    t = sub.add_parser("train", help="Train an LNBE model")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--samples", type=int, default=1000)

    args = parser.parse_args()
    if args.command == "process":
        cmd_process(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
