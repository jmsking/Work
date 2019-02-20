#! /usr/bin/python
# encoding:utf-8

import logging
import logging.config

logging.config.fileConfig('log_config.ini')
log = logging.getLogger('consoleOut')