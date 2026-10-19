import unittest
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from search_engine import search_db,query_terms,fts_words

class OfflineTests(unittest.TestCase):
    def test_normalization_and_stopwords(self):
        terms=query_terms('Как правильно сушить краску после нанесения?')
        self.assertTrue(terms)
        self.assertNotIn('как',terms)

    def test_no_injection_in_sql(self):
        self.assertEqual(search_db(BASE/'knowledge.sqlite', '" OR UNION SELECT *'),[])

    def test_retrieval(self):
        for question in ['просадка после сушки', 'подбор цвета серебристого металлика', 'инфракрасная лампа для сушки']:
            with self.subTest(question=question):
                found=search_db(BASE/'knowledge.sqlite',question)
                self.assertTrue(found)
                self.assertTrue(found[0]['content'])
                self.assertLessEqual(len(found),5)

    def test_foo_question(self):
        self.assertEqual(search_db(BASE/'knowledge.sqlite','Квантовая запутанность в космологии'),[])

if __name__=='__main__': unittest.main()
