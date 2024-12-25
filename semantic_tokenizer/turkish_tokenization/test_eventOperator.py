import unittest
from eventOperator import EventOperator

class TestEventOperator(unittest.TestCase):

    def setUp(self):
        self.event_operator = EventOperator()

    def test_consonant_hardening(self):
        self.assertEqual(self.event_operator.consonantHardening('kitap', 'ci'), 'kitapçi')
        self.assertEqual(self.event_operator.consonantHardening('ağaç', 'da'), 'ağaçta')
        self.assertEqual(self.event_operator.consonantHardening('çocuk', 'gi'), 'çocukki')
        self.assertEqual(self.event_operator.consonantHardening('ev', 'de'), 'evde')  # No hardening should occur

    def test_revert_consonant_hardening(self):
        self.assertEqual(self.event_operator.revertConsonantHardening('kitapçi', 'kitap'), ['kitap', 'ci'])
        self.assertEqual(self.event_operator.revertConsonantHardening('ağaçta', 'ağaç'), ['ağaç', 'da'])
        self.assertEqual(self.event_operator.revertConsonantHardening('çocukki', 'çocuk'), ['çocuk', 'gi'])
        self.assertEqual(self.event_operator.revertConsonantHardening('evde', 'ev'), 'ev')  # No hardening to revert

if __name__ == '__main__':
    unittest.main()