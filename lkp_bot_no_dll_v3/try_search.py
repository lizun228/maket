from search_engine import search_db
from pathlib import Path

p = Path(__file__).with_name('knowledge.sqlite')
print('ПРОВЕРКА БАЗЫ БЕЗ API И TELEGRAM')
while True:
    q = input('\nВопрос (Enter — выход): ').strip()
    if not q:
        break
    hits = search_db(p, q)
    if not hits:
        print('Совпадений нет.');continue
    for h in hits:
        print(f'\n[{h["thread_id"]}] №{h["start_message_id"]}-{h["end_message_id"]}  покрытие: {h["coverage"]:.0%}')
        print(h['content'][:800].replace('\n','\n   '))
