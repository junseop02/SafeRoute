"""Launch SafeRoute without storing or displaying the REST API key."""
import getpass,os,shutil,subprocess,sys,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
node=ROOT/'runtime/node.exe'
if not node.exists():
    system=shutil.which('node')
    if not system:raise SystemExit('Node.js is missing. Restore runtime/node.exe or install Node.js.')
    node=Path(system)
python=ROOT/'.venv/Scripts/python.exe'
if not python.exists():raise SystemExit('Run SETUP_SAFEROUTE.cmd first.')
env=os.environ.copy()
if not env.get('KAKAO_REST_API_KEY','').strip():
    if not sys.stdin.isatty():raise SystemExit('Run from an interactive terminal for hidden API key input.')
    with warnings.catch_warnings():
        warnings.simplefilter('error',getpass.GetPassWarning)
        try:key=getpass.getpass('Kakao REST API Key (hidden): ').strip()
        except getpass.GetPassWarning:raise SystemExit('Hidden input is unavailable. Use an interactive terminal.')
    if not key:raise SystemExit('No key entered.')
    env['KAKAO_REST_API_KEY']=key
print('SafeRoute is starting. Stop with Ctrl+C. The key is not saved to a file.')
child=subprocess.Popen([str(node),str(ROOT/'server.js')],cwd=ROOT,env=env)
try:sys.exit(child.wait())
except KeyboardInterrupt:
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:child.terminate();child.wait()
