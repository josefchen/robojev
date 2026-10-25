import argparse

from .robojev import RoboJev


def main() -> None:
    parser = argparse.ArgumentParser(description="Score whether a robot completed its task using a RoboJev vLLM server.")
    parser.add_argument("task", help="Task instruction given to the robot")
    parser.add_argument("attempt", nargs="+", help="One video file, or ten image files in time order")
    parser.add_argument("--server", default="http://localhost:8000", help="vLLM server URL (default: http://localhost:8000)")
    args = parser.parse_args()
    attempt = args.attempt[0] if len(args.attempt) == 1 else args.attempt
    print(f"{RoboJev(args.server).done(args.task, attempt):.3f}")


if __name__ == "__main__":
    main()
