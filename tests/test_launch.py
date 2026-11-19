import unittest,runpy,os,io,contextlib
from pathlib import Path
from unittest.mock import patch,MagicMock
ROOT=Path(__file__).resolve().parent.parent
class LaunchTests(unittest.TestCase):
 def test_hidden_key_not_logged_and_passed_only_to_child(self):
  child=MagicMock();child.wait.return_value=0;capture=io.StringIO();key='test-placeholder-not-a-real-key'
  with patch.dict(os.environ,{},clear=True),patch('pathlib.Path.exists',return_value=True),patch('sys.stdin.isatty',return_value=True),patch('getpass.getpass',return_value=key) as ask,patch('subprocess.Popen',return_value=child) as spawn,contextlib.redirect_stdout(capture):
   with self.assertRaises(SystemExit) as result:runpy.run_path(str(ROOT/'tools/launch.py'),run_name='__main__')
   self.assertEqual(result.exception.code,0);ask.assert_called_once();self.assertEqual(spawn.call_args.kwargs['env']['KAKAO_REST_API_KEY'],key)
  self.assertNotIn(key,capture.getvalue())
 def test_existing_key_reused_without_prompt(self):
  child=MagicMock();child.wait.return_value=0
  with patch.dict(os.environ,{'KAKAO_REST_API_KEY':'test-placeholder'},clear=True),patch('pathlib.Path.exists',return_value=True),patch('getpass.getpass') as ask,patch('subprocess.Popen',return_value=child),contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaises(SystemExit):runpy.run_path(str(ROOT/'tools/launch.py'),run_name='__main__')
   ask.assert_not_called()
 def test_missing_virtual_environment_does_not_start_node(self):
  with patch('pathlib.Path.exists',side_effect=lambda:False),patch('shutil.which',return_value=None),patch('subprocess.Popen') as spawn:
   with self.assertRaises(SystemExit):runpy.run_path(str(ROOT/'tools/launch.py'),run_name='__main__')
   spawn.assert_not_called()
if __name__=='__main__':unittest.main()
