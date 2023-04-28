from setuptools import setup, find_packages

setup(name='DIMVImputation',
      version= '0.1',
      description= 'Missing data imputation with Bayesian Conditional Expectation',
      url= 'http://github.com/maianhpuco/DIMV_imputation',
      author= 'Mai Anh Vu*, Thu Nguyen*, Tu T. Do, Nhan Phan, Pål Halvorsen, Michael A. Riegler, Binh T. Nguyen',
      author_email= 'maianhuel@gmail.com',
      packages= find_packages(),
      zip_safe=False)
