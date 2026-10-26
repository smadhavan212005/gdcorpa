import sys
import subprocess

def main():
    print("=== Running Full Proposed GD-CoRPA Model ===")
    cmd = [
        sys.executable, "train.py",
        "--config", "configs/default.yaml",
        "--ablation-mode", "full"
    ]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
