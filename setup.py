from setuptools import setup


setup(
    name='Automatic Speech Recognition Models',
    version='latest',
    description='End-to-End Speech Recognition models with PyTorch',
    author='Sangchun Ha',
    author_email='seomk9896@naver.com',
    url='https://github.com/hasangchun/Automatic-Speech-Recognition-Models',
    install_requires=[
        'torch>=1.4.0',
        'python-Levenshtein',
        'librosa >= 0.7.0',
        'numpy',
        'pandas',
        'hydra-core',
    ],
    python_requires='>=3',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)