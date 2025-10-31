from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nitty_gitty",
    version="0.1.0",
    author="jagan",
    author_email="jaganstudies06@gmail.com",
    description="A small deep learning framework for learning purpise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/learning_6m",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'cupy-cuda13x'
    ],
)