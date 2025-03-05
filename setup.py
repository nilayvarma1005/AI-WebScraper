from setuptools import setup, find_packages

setup(
    name="AI-WebScraper",
    version="1.0.0",
    author="Nilay Varma",
    author_email="nilayvarma.india@gmail.com",
    description="An intelligent AI-based web scraper",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nilayvarma1005/AI-WebScraper",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "aiohttp",
        "beautifulsoup4",
        "openai"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
