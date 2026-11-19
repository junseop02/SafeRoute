"""Create a local Python 3.12 environment. Run explicitly; downloads packages."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
if sys.version_info[:2] != (3,12):
    raise SystemExit('Python 3.12 is required. Install it and run SETUP_SAFEROUTE.cmd again.')
venv=ROOT/'.venv'
if not venv.exists():subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
python=venv/'Scripts/python.exe'
if not python.exists():raise SystemExit('Existing .venv is incomplete. Rename it and retry setup.')
subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements-lock.txt')],check=True)
subprocess.run([str(python),str(ROOT/'tools/check_environment.py')],check=True)
print('Setup complete. Start with START_SAFEROUTE.cmd.')
