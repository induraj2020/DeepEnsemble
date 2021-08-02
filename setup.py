import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='deepensemble',
     version='0.2',
     scripts=['DeepEnsemble'] ,
     author="Induraj P.Ramamurthy",
     author_email="induraj.gandhian@yahoo.com",
     description="An Wrapper for performing ensembling techniques on deep learning models",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/induraj2020/DeepEnsemble.git",
     packages=setuptools.find_packages(),
     classifiers=[
         'Development Status :: 3 - Alpha',
         'Intended Audience :: Science/Research',
         'License :: OSI Approved :: MIT License',
         'Natural Language :: English',
         'Operating System :: OS Independent',
         'Programming Language :: Python :: 3.7',
         'Topic :: Software Development :: Libraries :: Python Modules',
         'Topic :: Scientific/Engineering :: Artificial Intelligence',
     ],
    python_requires=">=3.6"
 )