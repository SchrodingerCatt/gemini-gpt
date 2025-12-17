#!/bin/bash
python3.12 -c "
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now run main.py
exec(open('main.py').read())
"
