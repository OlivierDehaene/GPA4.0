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
    package_data={
        'w2p.notebooks': [
            'stats_gp_prog.ipynb'
        ],
    },
    scripts=[
        'word2phonetic/utils/w2p-create_vocab',
        'word2phonetic/utils/w2p-decoder',
        'word2phonetic/utils/w2p-stats',
        'word2phonetic/utils/w2p-evaluate',
        'word2phonetic/utils/w2p-train_test_split',
        'word2phonetic/utils/w2p-export_model',
        'word2phonetic/utils/w2p-train',
    ],
    install_requires=[
        'tensor2tensor==1.6.2',
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
