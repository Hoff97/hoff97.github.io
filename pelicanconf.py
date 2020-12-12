#!/usr/bin/env python
# -*- coding: utf-8 -*- #

AUTHOR = 'Frithjof Winkelmann'
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
LINKS = ()

# Social widget
SOCIAL = (('github', 'https://github.com/Hoff97'),
          ('linkedin', 'https://www.linkedin.com/in/frithjof-winkelmann-338a99a8/'),)

DEFAULT_PAGINATION = 10

SUMMARY_MAX_LENGTH = 100

PLUGIN_PATHS = ['deps/']
PLUGINS = ['render_math', 'pelican-cite']

PUBLICATIONS_SRC = 'content/pubs.bib'

THEME = 'deps/Flex'

## Flex

SITENAME = "haskai"
SITETITLE = "Frithjof Winkelmann"
SITESUBTITLE = "Software Developer"
SITEDESCRIPTION = "Foo Bar's Thoughts and Writings"
SITELOGO = SITEURL + "/images/profile.png"
FAVICON = SITEURL + "/images/favicon.ico"

BROWSER_COLOR = "#333"
ROBOTS = "index, follow"

CC_LICENSE = {
    "name": "Creative Commons Attribution-ShareAlike",
    "version": "4.0",
    "slug": "by-sa"
}

COPYRIGHT_YEAR = 2020

MAIN_MENU = True
MENUITEMS = (("Categories", "/categories"), ("Tags", "/tags"),)

ADD_THIS_ID = "ra-77hh6723hhjd"
