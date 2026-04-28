import sys
import subprocess
from pathlib import Path

def main():
    backend_dir = Path(__file__).parent / "Backend"
    train_script = backend_dir / "train_model.py"
    
    if not train_script.exists():
        print(f"Error: {train_script} not found.")
        sys.exit(1)
        
    cmd = [sys.executable, str(train_script)] + sys.argv[1:]
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
