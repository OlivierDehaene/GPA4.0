from setuptools import find_packages
from setuptools import setup

setup(
    name='gpa4.0',
    version='1.0',
    description='Grapheme Phoneme Alignment',
    author='Olivier Dehaene',
    author_email='olivier.dehaene@gmail.com',
    url='https://github.com/OlivierDehaene/GPA4.0',
    license='Apache 2.0',
    packages=find_packages(),
    scripts=[
        'gpa/utils/gpa-create_vocab',
        'gpa/utils/gpa-decoder',
        'gpa/utils/gpa-stats',
        'gpa/utils/gpa-evaluate',
        'gpa/utils/gpa-train_test_split',
        'gpa/utils/gpa-export_model',
        'gpa/utils/gpa-train',
    ],
    install_requires=[
        'tensor2tensor==1.6.2',
        'tensorflow>=1.4.1',
        'pandas==0.20.1',
        'tqdm'
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
    keywords='grapheme 2 phoneme g2p gpa',
)
