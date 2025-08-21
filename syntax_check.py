import ast
import sys

def check_syntax(filename):
    try:
        with open(filename, 'r') as f:
            source = f.read()
        
        print(f"Reading file {filename}...")
        print("File contents:")
        print(source)
        print("\nChecking syntax...")
        
        tree = ast.parse(source)
        print("Syntax is valid!")
    except SyntaxError as e:
        print(f"Syntax error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    filename = r"c:\Users\Lenovo\PycharmProjects\GenAILC\rag_pipeline_recommendation_framework\rag_recommender\core\engine\recommendation.py"
    check_syntax(filename)
