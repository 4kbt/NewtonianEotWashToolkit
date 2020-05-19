import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="newt",
    version="0.0.1",
    author="John Greendeer Lee",
    author_email="jgl6@uw.edu",
    description="A newtonian gravity calculator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jglee6/PointGravity",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-v3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
