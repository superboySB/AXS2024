#!/bin/bash
set -x

isort .
yapf -prmi .
