from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines() 
    
setup(
    name='my_package',
    version='0.1.0',
    description='My awesome Python package',
    url='https://github.com/maianh.puco/DIMVImputation.git',
#     author='Your Name',
#     author_email='your@email.com',
    packages=find_packages(),
    package_dir={"": "src"}
    install_requires=requirements,
#     classifiers=[
#         'Development Status :: 3 - Alpha',
#         'Intended Audience :: Developers',
#         'Topic :: Software Development :: Libraries',
#         'License :: OSI Approved :: MIT License',
#         'Programming Language :: Python :: 3',
#         'Programming Language :: Python :: 3.6',
#         'Programming Language :: Python :: 3.7',
#         'Programming Language :: Python :: 3.8',
#         'Programming Language :: Python :: 3.9',
#     ],
    keywords='python package',
    python_requires='>=3.6',
) 