#!/bin/sh
# Kept so that the old command still works: the installation is now `./text2kg.sh install`.
exec "$(dirname "$0")/text2kg.sh" install "$@"