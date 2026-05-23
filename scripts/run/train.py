import os
import logging
import argparse


from .run_from_files import run

logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train policy on given environment using specified algorithm."
    )
    parser.add_argument(
        "--algo", default="hits", help="Which algorithm to use (hac or hits)."
    )
    parser.add_argument(
        "--env",
        default="Pendulum",
        help="Which environment to run (Platforms, Drawbridge or Tennis2D).",
    )

    args = parser.parse_args()

    assert args.algo in {"hits", "hits_no_budget", "hac", "sac"}
    assert args.env in {
        "AntFourRooms",
        "Drawbridge",
        "Pendulum",
        "Platforms",
        "Tennis2D",
        "UR5Reacher",
    }

    LOG.info(f"Training with {args.algo} on {args.env} environment.")

    path = os.path.join("./data", args.env, args.algo + "_trained")
    LOG.info(f"Dir path: {path}")
    run(path)
