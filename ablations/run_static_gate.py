import sys
import subprocess

def main():
    print("=== Running Static Gate Ablation ===")
    cmd = [
        sys.executable, "train.py",
        "--config", "configs/default.yaml",
        "--ablation-mode", "static_gate"
    ]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
