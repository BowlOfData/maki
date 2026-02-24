from setuptools import setup, find_packages

setup(
    name="maki",
    version="0.1.0",
    author="Bowl of Data",
    author_email="bowlofdata@gmail.com",
    description="A Python framework for multi-agent LLM interactions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bowlofdata/maki",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "maki-example=maki.examples.agent_example:main",
        ],
    },
)