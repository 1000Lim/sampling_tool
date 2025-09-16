import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sampling_tool",  # Replace with your own library name
    version="1.0.0",
    author="1000lim",
    author_email="lin.qian@stradvision.com",
    description="Sampling tool for Collecting Rawdata.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.stradvision.com/StradVision/strada/tree/main/data/ingestion/sampling",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'annotated-types==0.7.0',
        'loguru==0.7.2',
        'numpy==2.1.1',
        'opencv-python==4.10.0.84',
        'pydantic==2.9.2',
        'pypcd4==1.1.0',
        'python-lzf==0.2.6',
        'tqdm==4.66.5',
        'typing_extensions==4.12.2',
        'xmltodict==0.13.0',
    ],
)
