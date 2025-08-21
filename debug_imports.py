import sys
import traceback

def import_and_print(module_name):
    try:
        print(f"Importing {module_name}...")
        __import__(module_name)
        print(f"Successfully imported {module_name}")
    except Exception as e:
        print(f"Error importing {module_name}:")
        traceback.print_exc()
        
print("Python path:")
for p in sys.path:
    print(p)

modules = [
    "rag_recommender",
    "rag_recommender.api",
    "rag_recommender.api.main",
    "rag_recommender.core",
    "rag_recommender.core.engine",
    "rag_recommender.core.engine.recommendation",
    "rag_recommender.core.models",
    "rag_recommender.core.models.base"
]

for module in modules:
    import_and_print(module)
