from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# This setup file assumes you have RENAMED your main package folder from 'monkey3' to 'm3',
# and your main script from 'monkey3.py' to 'main.py' (i.e., m3/main.py).
setup(
    name='m3',
    version='1.0.0',
    author='John Chen',
    author_email='your_email@example.com',
    description='A tool for qualitative data analysis powered by local LLMs.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # find_packages() will automatically discover the 'm3' package
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'llama-index',
        'llama-index-llms-ollama',
        'llama-index-embeddings-ollama',
        'llama-index-vector-stores-chroma',
        'chromadb',
        'PyYAML'  # Add PyYAML for YAML file handling
    ],
    # The entry point now points to the cli function inside the 'm3' package's 'main.py' file.
    entry_points='''
        [console_scripts]
        m3=m3.main:cli
    '''
)