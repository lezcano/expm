import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-expm", # Replace with your own username
    version="0.0.1",
    author="Mario Lezcano-Casado",
    author_email="mario.lezcanocasado@maths.ox.ac.uk",
    description="Two differentiable implementations of the exponential of matrices in Pytorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lezcano/expm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
