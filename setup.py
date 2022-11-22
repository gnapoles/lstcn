from setuptools import setup

setup(
    name='lstcn',
    version='0.1.0',
    description='Long Short-term Cognitive Networks',
    long_description="Long Short-term Cognitive Networks (LSTCNs) are a type of recurrent neural network for time series forecasting. "
    + "LSTCNs use a fast learning algorithm supporting univariate and multivariate time series.",
    url='https://github.com/gnapoles/lstcn',
    author='Gonzalo Nápoles',
    author_email='g.r.napoles@uvt.nl',
    license='Apache License 2.0',
    packages=['lstcn'],
    install_requires=['hampel',
                      'numpy', 'scipy', 'scikit-learn', 'pandas',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.8',
    ],
)