import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from spear_rag.rag.geo_parser import SpatialQuery
from spear_rag.rag.answer_generator import generate_answer
from spear_rag.rag.context_builder import build_context

q = SpatialQuery(raw_query="What is the land cover?", location="Punjab", bbox=(29.5, 32.5, 73.9, 76.7))
results = {"s2": {"n_retrieved": 47173, "ndvi": 0.06, "ndwi": -0.04}}
ctx = build_context(q, results)
ans = generate_answer(q, results, ctx)
print("ANSWER DICT:")
print(ans["answer"])
