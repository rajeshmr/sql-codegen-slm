from setuptools import setup, find_packages

setup(
    name="sql_codegen",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A fine-tuned Mistral-7B model for SQL query generation from natural language",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sql-codegen-slm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10,<3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
