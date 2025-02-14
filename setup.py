from setuptools import setup, find_packages

setup(
    name="object_contradictions",
    version="0.1",
    author="Marie Freischlad",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "tqdm",
        "torch",
        "openai",
        "tiktoken",
        "pandas",
        "stanza",
        "spacy",
        "thefuzz",
    ],
)
