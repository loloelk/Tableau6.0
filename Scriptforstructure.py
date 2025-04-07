import os
import argparse
import re
import sys
import ast
import json
from collections import defaultdict

# --- Configuration ---

# Regex patterns for non-Python files
PATTERNS = {
    'javascript': {
        'class': re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:\{|extends|implements)", re.MULTILINE),
        'function': re.compile(r"^\s*(?:async\s+)?function\s*\*?\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", re.MULTILINE),
        'import': re.compile(r"^\s*import\s+(?:.*?from\s+)?['\"]([^'\"]+)['\"]", re.MULTILINE),
        'require': re.compile(r"(?:const|let|var)\s+.*?=\s*require\(['\"]([^'\"]+)['\"]\)", re.MULTILINE)
    },
    'html': {
        'component': re.compile(r"<([a-zA-Z][a-zA-Z0-9_-]*)(?:\s+[^>]*)?(?:/>|>.*?</\1>)", re.MULTILINE | re.DOTALL),
    },
    'css': {
        'selector': re.compile(r"([^\{\}]+)\s*\{[^\}]*\}", re.MULTILINE),
    },
    'yaml': {
        'key': re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*):", re.MULTILINE),
    }
}

# Map extensions to language pattern keys (excluding Python)
EXT_TO_LANG = {
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'javascript',
    '.tsx': 'javascript',
    '.html': 'html',
    '.css': 'css',
    '.scss': 'css',
    '.yaml': 'yaml',
    '.yml': 'yaml',
}

# Framework detection patterns
FRAMEWORK_PATTERNS = {
    'dash': [
        r'import\s+dash',
        r'from\s+dash',
        r'dcc\.Graph',
        r'app\.layout',
        r'@app\.callback'
    ],
    'flask': [
        r'from\s+flask\s+import',
        r'Flask\(',
        r'@app\.route'
    ],
    'django': [
        r'from\s+django',
        r'urlpatterns',
        r'class\s+\w+\(.*?Model\)',
        r'class\s+\w+\(.*?View\)'
    ],
    'streamlit': [
        r'import\s+streamlit',
        r'st\.'
    ],
    'pandas': [
        r'import\s+pandas',
        r'from\s+pandas',
        r'pd\.'
    ],
    'numpy': [
        r'import\s+numpy',
        r'from\s+numpy',
        r'np\.'
    ],
    'plotly': [
        r'import\s+plotly',
        r'from\s+plotly',
        r'px\.',
        r'go\.'
    ],
    'matplotlib': [
        r'import\s+matplotlib',
        r'from\s+matplotlib',
        r'plt\.'
    ],
    'tensorflow': [
        r'import\s+tensorflow',
        r'from\s+tensorflow',
        r'tf\.'
    ],
    'pytorch': [
        r'import\s+torch',
        r'from\s+torch',
        r'nn\.'
    ],
    'scikit-learn': [
        r'from\s+sklearn',
        r'import\s+sklearn'
    ]
}

# Special file analyzers
SPECIAL_FILE_PATTERNS = {
    'requirements.txt': re.compile(r"^([A-Za-z0-9_\-\.]+)(?:[=<>]+.*)?$", re.MULTILINE),
    'dockerfile': re.compile(r"^(FROM|RUN|COPY|ADD|ENTRYPOINT|CMD|EXPOSE|ENV)\s+", re.MULTILINE),
}

# Default directories to ignore
DEFAULT_IGNORE_DIRS = {
    '.git', 'node_modules', 'venv', '.venv', 'env', '__pycache__',
    'dist', 'build', 'coverage', '.vscode', '.idea', 'target', 'logs'
}

# Default file extensions to analyze
PYTHON_EXT = {'.py'}
DEFAULT_ANALYZE_EXT_REGEX = set(EXT_TO_LANG.keys())
DEFAULT_ANALYZE_EXT = PYTHON_EXT | DEFAULT_ANALYZE_EXT_REGEX | {
     '.java', '.cs', '.go', '.php', '.rb', '.json', '.md'
}

# --- AST Visitor for Python Code Analysis ---

class PythonAstVisitor(ast.NodeVisitor):
    """
    Visits AST nodes to find functions, classes, imports, and variables in Python code.
    """
    def __init__(self):
        self.functions = set()
        self.classes = set()
        self.imports = set()
        self.variables = set()
        # Track framework-specific components
        self.dash_components = set()
        self.flask_routes = set()
        
    def visit_FunctionDef(self, node):
        self.functions.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.functions.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module_name = node.module if node.module else "." * node.level
        for alias in node.names:
            if module_name:
                self.imports.add(module_name)
            self.imports.add(alias.name)
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        # Capture variable assignments
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.add(target.id)
                
                # Check for Dash layout assignment
                if target.id == 'layout' and isinstance(node.value, ast.Call):
                    if hasattr(node.value, 'func') and hasattr(node.value.func, 'attr'):
                        if node.value.func.attr in ['Div', 'Graph', 'Layout']:
                            self.dash_components.add('layout')
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Try to detect framework components in function calls
        if isinstance(node.func, ast.Attribute):
            # Check for Dash components
            if hasattr(node.func, 'value') and hasattr(node.func.value, 'id'):
                if node.func.value.id in ['dcc', 'html', 'dash_bootstrap_components', 'dbc']:
                    self.dash_components.add(f"{node.func.value.id}.{node.func.attr}")
                elif node.func.value.id == 'app' and node.func.attr == 'callback':
                    self.dash_components.add('app.callback')
        
        self.generic_visit(node)
    
    def visit_Decorator(self, node):
        # Check for Flask routes
        if isinstance(node.func, ast.Attribute):
            if hasattr(node.func, 'value') and hasattr(node.func.value, 'id'):
                if node.func.value.id == 'app' and node.func.attr == 'route':
                    self.flask_routes.add('app.route')
        
        self.generic_visit(node)

# --- Helper Functions ---

def analyze_python_ast(filepath):
    """
    Analyzes a Python file using the AST module for accurate parsing.
    """
    extracted_info = {
        'function': set(), 
        'class': set(), 
        'import': set(),
        'variable': set(),
        'framework_component': set()
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=filepath)
        visitor = PythonAstVisitor()
        visitor.visit(tree)
        
        extracted_info['function'] = visitor.functions
        extracted_info['class'] = visitor.classes
        extracted_info['import'] = visitor.imports
        extracted_info['variable'] = visitor.variables
        
        # Add framework-specific components if found
        if visitor.dash_components:
            extracted_info['framework_component'] = visitor.dash_components
        elif visitor.flask_routes:
            extracted_info['framework_component'] = visitor.flask_routes

    except SyntaxError as e:
        print(f"  - Warning: Skipping analysis due to Python SyntaxError in {filepath}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  - Warning: Could not read or analyze file {filepath} with AST: {e}", file=sys.stderr)
        return None

    # Convert sets back to sorted lists for consistent output
    for key in extracted_info:
        extracted_info[key] = sorted(list(extracted_info[key]))

    return extracted_info


def get_language_patterns_regex(file_extension):
    """
    Gets the regex patterns for a given file extension (for non-Python).
    """
    lang = EXT_TO_LANG.get(file_extension.lower())
    if lang:
        return PATTERNS.get(lang)
    return None


def analyze_file_content_regex(filepath, patterns):
    """
    Reads file content and extracts info based on regex patterns (for non-Python).
    """
    extracted_info = {key: set() for key in patterns}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for key, regex in patterns.items():
                matches = regex.finditer(content)
                for match in matches:
                    found_name = next((g for g in match.groups() if g is not None), None)
                    if found_name:
                        if key == 'require':
                            found_name = os.path.basename(found_name)
                        extracted_info[key].add(found_name.strip())
    except Exception as e:
        print(f"  - Warning: Could not read or analyze file {filepath} with regex: {e}", file=sys.stderr)
        return None
    
    for key in extracted_info:
        extracted_info[key] = sorted(list(extracted_info[key]))
    
    return extracted_info


def analyze_special_file(filepath, filename):
    """
    Analyze special files like requirements.txt or package.json.
    """
    extracted_info = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            if filename.lower() == 'requirements.txt':
                matches = SPECIAL_FILE_PATTERNS['requirements.txt'].finditer(content)
                packages = [match.group(1) for match in matches if match.group(1)]
                extracted_info['packages'] = packages
            
            elif filename.lower() == 'package.json':
                try:
                    data = json.loads(content)
                    dependencies = list(data.get('dependencies', {}).keys())
                    devDependencies = list(data.get('devDependencies', {}).keys())
                    if dependencies:
                        extracted_info['dependencies'] = dependencies
                    if devDependencies:
                        extracted_info['devDependencies'] = devDependencies
                except json.JSONDecodeError:
                    pass
            
            elif filename.lower() == 'dockerfile':
                matches = SPECIAL_FILE_PATTERNS['dockerfile'].finditer(content)
                commands = [match.group(1) for match in matches if match.group(1)]
                extracted_info['commands'] = commands
                
            elif filename.lower().endswith(('.yaml', '.yml')) and 'config' in filename.lower():
                try:
                    import yaml
                    data = yaml.safe_load(content)
                    if isinstance(data, dict):
                        extracted_info['configuration'] = list(data.keys())
                except Exception:
                    # If yaml module is not available or parsing fails, don't attempt further analysis
                    pass
                
    except Exception as e:
        print(f"  - Warning: Could not analyze special file {filepath}: {e}", file=sys.stderr)
        return None
    
    return extracted_info


def detect_frameworks(python_file_contents):
    """
    Detect frameworks and libraries used in the project based on imports and patterns.
    
    Args:
        python_file_contents: Dictionary mapping file paths to their contents.
    
    Returns:
        Dictionary mapping framework names to their confidence scores (number of matches).
    """
    framework_matches = defaultdict(int)
    
    for filepath, content in python_file_contents.items():
        for framework, patterns in FRAMEWORK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.MULTILINE):
                    framework_matches[framework] += 1
    
    # Filter out frameworks with very low match counts
    return {k: v for k, v in framework_matches.items() if v >= 2}


def categorize_files(all_files, frameworks):
    """
    Categorize files based on their purpose and detected frameworks.
    
    Args:
        all_files: List of all file paths in the project.
        frameworks: List of detected frameworks.
    
    Returns:
        Dictionary of file categories.
    """
    categories = {
        'entrypoints': [],
        'dashboard_components': [],
        'config_files': [],
        'data_files': [],
        'test_files': [],
        'documentation': [],
        'utility_files': []
    }
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Entrypoints
        if filename in ('app.py', 'main.py', 'server.py', 'run.py', 'manage.py'):
            categories['entrypoints'].append(filepath)
        
        # Dashboard components (if using Dash)
        elif 'dash' in frameworks and any(x in filename for x in ('dashboard', 'chart', 'graph', 'plot', 'component')):
            categories['dashboard_components'].append(filepath)
        
        # Config files
        elif any(filename.endswith(ext) for ext in ('.json', '.yaml', '.yml', '.env', '.ini', '.conf')) and (
            'config' in filename or 'settings' in filename):
            categories['config_files'].append(filepath)
        
        # Data files
        elif any(filename.endswith(ext) for ext in ('.csv', '.json', '.xlsx', '.xls', '.db', '.sqlite')):
            categories['data_files'].append(filepath)
        
        # Test files
        elif filename.startswith('test_') or filename.endswith('_test.py') or 'tests' in filepath:
            categories['test_files'].append(filepath)
        
        # Documentation
        elif filename.endswith(('.md', '.rst', '.txt')) or filename in ('README', 'LICENSE', 'CONTRIBUTING'):
            categories['documentation'].append(filepath)
        
        # Utilities
        elif any(keyword in filename for keyword in ('util', 'helper', 'common', 'tools', 'logging')):
            categories['utility_files'].append(filepath)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


# --- Main Logic ---

def analyze_codebase(root_dir, output_file, ignore_dirs, analyze_ext_all, debug=False):
    """
    Analyzes the codebase structure, using AST for Python and regex for others.
    
    Args:
        root_dir: The root directory of the codebase.
        output_file: Path to the output Markdown file.
        ignore_dirs: Set of directory names to ignore.
        analyze_ext_all: Set of file extensions to analyze.
        debug: Whether to print debug information.
    """
    root_dir = os.path.abspath(root_dir)
    analyze_ext_py = PYTHON_EXT.intersection(analyze_ext_all)
    analyze_ext_regex = DEFAULT_ANALYZE_EXT_REGEX.intersection(analyze_ext_all)

    print(f"Starting analysis of: {root_dir}")
    print(f"Ignoring directories: {', '.join(sorted(list(ignore_dirs)))}")
    print(f"Analyzing Python files with AST: {', '.join(sorted(list(analyze_ext_py)))}")
    print(f"Analyzing other files with regex: {', '.join(sorted(list(analyze_ext_regex)))}")
    print(f"Output will be written to: {output_file}")
    
    # Store all file paths for categorization
    all_files = []
    
    # Store Python file contents for framework detection
    python_file_contents = {}
    
    # First pass - collect files and Python contents
    for current_root, dirs, files in os.walk(root_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in files:
            file_path = os.path.join(current_root, filename)
            relative_path = os.path.relpath(file_path, root_dir)
            all_files.append(relative_path)
            
            # Collect Python file contents for framework detection
            _, file_extension = os.path.splitext(filename)
            if file_extension.lower() in analyze_ext_py:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        python_file_contents[relative_path] = f.read()
                except Exception:
                    pass

    # Detect frameworks
    frameworks = detect_frameworks(python_file_contents)
    detected_frameworks = list(frameworks.keys())
    
    if detected_frameworks:
        print(f"Detected frameworks/libraries: {', '.join(detected_frameworks)}")
    else:
        print("No specific frameworks detected.")
    
    # Categorize files
    file_categories = categorize_files(all_files, detected_frameworks)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # --- Write Header and Framework Detection ---
            outfile.write(f"# Codebase Structure Analysis: {os.path.basename(root_dir)}\n\n")
            outfile.write(f"Root Directory: `{root_dir}`\n\n")
            
            if detected_frameworks:
                outfile.write("## Detected Frameworks & Libraries\n\n")
                for framework in sorted(frameworks.items(), key=lambda x: x[1], reverse=True):
                    outfile.write(f"- **{framework[0]}** (confidence: {framework[1]})\n")
                outfile.write("\n")
            
            # --- Write File Structure ---
            outfile.write("## File Structure\n\n")
            
            for current_root, dirs, files in os.walk(root_dir, topdown=True):
                dirs[:] = [d for d in dirs if d not in ignore_dirs]
                relative_root = os.path.relpath(current_root, root_dir)
                if relative_root == '.':
                    relative_root = ""
                depth = relative_root.count(os.sep)
                indent = "  " * depth

                if relative_root:
                    outfile.write(f"{indent}📁 **{os.path.basename(current_root)}/**\n")
                elif depth == 0 and not relative_root:
                    indent = ""

                file_indent = indent + "  "
                sorted_files = sorted(files)
                for filename in sorted_files:
                    file_path = os.path.join(current_root, filename)
                    relative_path = os.path.relpath(file_path, root_dir)
                    _, file_extension = os.path.splitext(filename)
                    ext_lower = file_extension.lower()

                    outfile.write(f"{file_indent}📄 {filename}\n")
                    
                    extracted = None
                    analysis_type = ""

                    # --- Special File Analysis ---
                    if filename.lower() in ('requirements.txt', 'package.json', 'dockerfile') or (
                        filename.lower().endswith(('.yaml', '.yml')) and 'config' in filename.lower()):
                        analysis_indent = file_indent + "  "
                        special_info = analyze_special_file(file_path, filename)
                        if special_info:
                            outfile.write(f"{analysis_indent}*Special File Analysis:*\n")
                            for key, items in special_info.items():
                                if items:
                                    # Limit long lists to improve readability
                                    display_items = items
                                    if len(items) > 10:
                                        display_items = items[:10] + ['...']
                                    outfile.write(f"{analysis_indent}  - {key.capitalize()}: `{', '.join(display_items)}`\n")
                    
                    # --- Python Analysis with AST ---
                    elif ext_lower in analyze_ext_py:
                        extracted = analyze_python_ast(file_path)
                        analysis_type = "AST"
                    
                    # --- Other Languages with Regex ---
                    elif ext_lower in analyze_ext_regex:
                        patterns = get_language_patterns_regex(ext_lower)
                        if patterns:
                            extracted = analyze_file_content_regex(file_path, patterns)
                            analysis_type = "Regex"

                    # --- Write Analysis Results ---
                    if analysis_type:
                        analysis_indent = file_indent + "  "
                        if extracted:
                            outfile.write(f"{analysis_indent}*Analysis ({analysis_type}):*\n")
                            empty_analysis = True
                            for key, items in extracted.items():
                                if items:
                                    empty_analysis = False
                                    # Limit long import lists for readability
                                    display_items = items
                                    if key == 'import' and len(items) > 10:
                                        display_items = items[:10] + ['...']
                                    outfile.write(f"{analysis_indent}  - {key.capitalize()}s: `{', '.join(display_items)}`\n")
                            if empty_analysis:
                                outfile.write(f"{analysis_indent}  - (No key elements found)\n")
                        elif extracted is None:
                            outfile.write(f"{analysis_indent}*Analysis ({analysis_type}): (Could not read/analyze file or Syntax Error)*\n")
            
            # --- Write Summary Section ---
            outfile.write("\n## Project Summary\n\n")
            
            # File categorization
            for category, files in file_categories.items():
                # Use proper title for category
                category_title = ' '.join(word.capitalize() for word in category.split('_'))
                outfile.write(f"### {category_title}\n\n")
                for file in sorted(files):
                    outfile.write(f"- `{file}`\n")
                outfile.write("\n")
            
            # Project complexity metrics
            outfile.write("### Project Complexity\n\n")
            outfile.write(f"- Total Files: {len(all_files)}\n")
            outfile.write(f"- Python Files: {len([f for f in all_files if f.endswith('.py')])}\n")
            if 'data_files' in file_categories:
                outfile.write(f"- Data Files: {len(file_categories['data_files'])}\n")
            outfile.write("\n")

            outfile.write("\n---\nAnalysis Complete.\n")
        print(f"\nAnalysis complete. Structure written to {output_file}")

    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}", file=sys.stderr)


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced codebase analyzer using AST for Python and regex for other languages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:
  python codebase_analyzer.py /path/to/your/codebase project_analysis.md
  python codebase_analyzer.py . current_project_analysis.md --ignore build --ignore .cache
  python codebase_analyzer.py ../my_project analysis.md --ext .py --ext .js --debug

Features:
- Uses Python's AST module for accurate Python code analysis
- Uses regex patterns for other languages
- Detects frameworks and libraries used in the project
- Categorizes files by their purpose
- Creates a comprehensive project summary
- Handles special files like requirements.txt and package.json
"""
    )
    parser.add_argument(
        "codebase_path",
        help="Path to the root directory of the codebase to analyze."
    )
    parser.add_argument(
        "output_file",
        help="Path to the output Markdown (.md) file to be created."
    )
    parser.add_argument(
        "--ignore",
        action="append",
        dest="ignore_dirs_list",
        default=[],
        help="Specify a directory name to ignore (e.g., --ignore build). Use multiple times."
    )
    parser.add_argument(
        "--ext",
        action="append",
        dest="analyze_extensions_list",
        default=[],
        help="Specify a file extension to analyze (e.g., --ext .py). Use multiple times."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for troubleshooting."
    )

    args = parser.parse_args()

    # Setup ignore directories and analyze extensions
    final_ignore_dirs = DEFAULT_IGNORE_DIRS.copy()
    final_ignore_dirs.update(args.ignore_dirs_list)

    final_analyze_ext_all = DEFAULT_ANALYZE_EXT.copy()
    if args.analyze_extensions_list:
        final_analyze_ext_all = set()
        for ext in args.analyze_extensions_list:
            if not ext.startswith('.'):
                final_analyze_ext_all.add('.' + ext.lower())
            else:
                final_analyze_ext_all.add(ext.lower())
        print(f"Overriding default extensions. Analyzing ONLY: {', '.join(sorted(list(final_analyze_ext_all)))}")

    # Validate codebase_path
    if not os.path.isdir(args.codebase_path):
        print(f"Error: Codebase path '{args.codebase_path}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    analyze_codebase(args.codebase_path, args.output_file, final_ignore_dirs, final_analyze_ext_all, args.debug)