import sys

from .robojev import RoboJev


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: robojev 'task' attempt.mp4 | frame1.jpg ... (start to end, as many as decision.json says)")
    attempt = sys.argv[2] if len(sys.argv) == 3 else sys.argv[2:]
    print(f"{RoboJev().done(sys.argv[1], attempt):.3f}")


if __name__ == "__main__":
    main()
