import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class ProviderCodeTests(unittest.TestCase):
    def test_supported_backends(self):
        code=(ROOT/'bot.py').read_text(encoding='utf-8')
        for fragment in ('AI_PROVIDER', "'proxyapi'", "'ollama'", "'offline'", 'generate_response'):
            self.assertIn(fragment,code)
    def test_no_auth_key_in_example(self):
        env=(ROOT/'.env.example').read_text(encoding='utf-8')
        self.assertIn('AI_API_KEY=\n',env)

if __name__=='__main__': unittest.main()
