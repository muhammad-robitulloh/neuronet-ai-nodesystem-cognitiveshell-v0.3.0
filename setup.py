from setuptools import setup, find_packages
from pathlib import Path
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name='neuronet-ai-cognitiveshell',
    version='0.2.6',
    packages=find_packages(),
    install_requires=[
    "python-telegram-bot>=20.0",
    "python-dotenv>=0.21.0",
    "pexpect>=4.8",
    "requests>=2.25",
    "pytz>=2022.1",
    "python-telegram-bot[jobqueue]"
    ],
    entry_points={
        'console_scripts': [
            'cognitiveshell=cognitive_shell.cognitiveshell:main',
            'quickstart=cognitive_shell.quick_start:main',
        ],
    },
    author='Muhammad Robitulloh',
    author_email='muhammadrobitulloh19@gmail.com',
    description='Semi-Agentic Terminal AI assistant with shell error detection and debugging via Telegram',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.7',
)
