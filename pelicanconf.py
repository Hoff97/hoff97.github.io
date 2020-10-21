#!/usr/bin/env python
# -*- coding: utf-8 -*- #

AUTHOR = 'Frithjof Winkelmann'
SITENAME = 'haskai'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Europe/Berlin'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'https://getpelican.com/'),)

# Social widget
SOCIAL = (('Github', 'https://github.com/Hoff97'),
          ('LinkedIn', 'https://www.linkedin.com/in/frithjof-winkelmann-338a99a8/'),)

DEFAULT_PAGINATION = 10

SUMMARY_MAX_LENGTH = 100

PLUGIN_PATHS = ['deps/']
PLUGINS = ['render_math', 'pelican-cite']

PUBLICATIONS_SRC = 'content/pubs.bib'

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True