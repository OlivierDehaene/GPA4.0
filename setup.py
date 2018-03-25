from setuptools import find_packages
from setuptools import setup

setup(
    name='w2p',
    version='1.0',
    description='Word To Phonetic',
    author='Olivier Dehaene',
    author_email='olivier.dehaene@gmail.com',
    url='https://github.com/OlivierDehaene/word2phonetic',
    license='Apache 2.0',
    packages=find_packages(),
    scripts=[
        'word2phonetic/utils/create_vocab.py',
        'word2phonetic/utils/dataset_train_test_split.py',
        'word2phonetic/utils/export_model.py'
    ],
    install_requires=[
        'tensor2tensor==1.5.5',
        'tensorflow>=1.4.1',
        'pandas'
    ],
    extras_require={
        'tensorflow_gpu': ['tensorflow-gpu>=1.4.1']
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Neuroscience',
    ],
    keywords='grapheme 2 phoneme g2p',
)