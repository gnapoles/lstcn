from setuptools import setup

setup(
    name='lstcn',
    version='0.1.0',
    description='Long Short-term Cognitive Networks',
    long_description="The Long Short-term Cognitive Network (LSTCN) model is an efficient recurrent neural network "
                     + "for time series forecasting. It supports both one-step-ahead and multiple-step-ahead "
                     + "forecasting of univariate and multivariate time series. The LSTCN model is competitive "
                     + "compared to state-of-the-art recurrent neural networks such as LSTM and GRU in terms of "
                     + "forecasting error while being much faster.",
    url='https://github.com/gnapoles/lstcn',
    author='Gonzalo Nápoles',
    author_email='g.r.napoles@uvt.nl',
    license='Apache License 2.0',
    packages=['lstcn'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.8',
    ],
)