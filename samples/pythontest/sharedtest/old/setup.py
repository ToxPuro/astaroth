from distutils.core import setup, Extension

module1 = Extension('justtest',
                    sources = ['justtestmodule.c'])

setup (name = 'mytest',
       version = '1.0',
       description = 'This is a test',
       ext_modules = [module1])


