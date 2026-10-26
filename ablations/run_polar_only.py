import sys
import subprocess

def main():
    print("=== Running Polar-Only Ablation ===")
    cmd = [
        sys.executable, "train.py",
        "--config", "configs/default.yaml",
        "--ablation-mode", "polar_only"
    ]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
