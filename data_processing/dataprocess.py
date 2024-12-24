import pandas as pd
import re
from tree_sitter import Language, Parser
import sys

# Increase recursion limit
sys.setrecursionlimit(10000)

# Load the Reveal dataset
file_path = 'data/Devign_utf8.csv'  # 修改为你文件的路径
devign_data = pd.read_csv(file_path)


C_LANGUAGE = Language('build/my-languages.so', 'c')

# Create parser instance
parser = Parser()
parser.set_language(C_LANGUAGE)

# Function to extract comments
def extract_comments(code):
    """
    Use regex to extract comments from C code.
    Supports // and /* ... */ comment formats.
    """
    pattern = r'//.*|/\*[\s\S]*?\*/'
    comments = re.findall(pattern, code)
    return ' '.join(comments) if comments else ''


# Function to walk through the AST tree iteratively to avoid deep recursion
def walk_tree_iteratively(node):
    """
    Use iterative traversal to avoid deep recursion when walking the AST.
    """
    stack = [node]
    tree_structure = []

    while stack:
        current_node = stack.pop()
        node_info = {
            'type': current_node.type,
            'start': current_node.start_point,
            'end': current_node.end_point,
            'children': []
        }

        # Add children to the stack for further processing
        if current_node.children:
            stack.extend(current_node.children)

        tree_structure.append(node_info)

    return tree_structure


# Set maximum code size to avoid stack overflow on large code blocks
MAX_CODE_SIZE = 10000  # 设置代码字符数的最大阈值


# Function to generate AST for each code snippet
def generate_ast_tree(code, parser):
    """
    Walk through the AST using tree-sitter and avoid deep recursion.
    """
    if len(code) > MAX_CODE_SIZE:
        return "Code too large to process"

    try:
        tree = parser.parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        return walk_tree_iteratively(root_node)
    except RecursionError:
        return "AST generation failed: RecursionError"
    except Exception as e:
        return "AST generation failed: General Error"


# Process the Reveal dataset to extract comments and AST
devign_data['comments'] = devign_data['func'].apply(extract_comments)
devign_data['ast'] = devign_data['func'].apply(lambda code: generate_ast_tree(code, parser))

# Save the processed dataset with AST and comments into a new CSV file
output_file_path = 'devign_processed.csv'
devign_data[['func', 'comments', 'ast', 'target']].to_csv(output_file_path, index=False)

print(f"Processed dataset saved to {output_file_path}")