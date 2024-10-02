
from meta_ai_api import MetaAI
# from pydantic import ClassVar
import time

meta = MetaAI()
for i in range(100):
    print(i)
    print(meta.prompt("dsafdsafadsfasdfdsfdsafdsaf"))