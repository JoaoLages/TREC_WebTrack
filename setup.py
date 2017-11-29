from setuptools import setup, find_packages

try:
    from pip.req import parse_requirements
    import pip.download

    # parse_requirements() returns generator of pip.req.InstallRequirement objects
    install_reqs = parse_requirements(
        "requirements.txt", session=pip.download.PipSession()
    )
    # install_requires is a list of requirement
    install_requires = [str(ir.req) for ir in install_reqs]
except:
    # This is a bit of an ugly hack, but pip is not installed on EMR
    install_requires = []


package_data = {
}

setup(
    name='TREC_WebTrack',
    version='0.0.1',
    py_modules=['TREC_WebTrack'],
    # test_suite='tests',
    # See: http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    install_requires=install_requires,
    package_data=package_data,
)
