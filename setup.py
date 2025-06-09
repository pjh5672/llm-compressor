import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LLM-Compressor",
    version=os.getenv("GITHUB_REF_NAME", "v0.0.1"),
    author="Jiho Park",
    author_email="pjh5672.dev@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(exclude=["*tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=['wheel'],
    install_requires=[
        "torch==2.7.0", 
        "transformers==4.52.4", 
        "lm_eval==0.4.8",
        "accelerate==1.7.0",
    ],
    python_requires='>=3.12',
)