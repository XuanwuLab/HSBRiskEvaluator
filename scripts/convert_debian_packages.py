#! /usr/bin/env python3

import yaml
from hsbriskevaluator.utils.file import get_data_dir
with open(get_data_dir()/ "debian" / "packages.yaml", "r") as f:
    packages=yaml.safe_load(f)

newpkgs = {'packages': []}
for pkg_name in packages.keys():
    pkg=packages[pkg_name]
    pkg['labels'] = []
    if pkg['priority'] in ('required', 'important', 'standard'):
        pkg['labels'].append(f"priority:{pkg['priority']}")
    if pkg['essential']:
        pkg['labels'].append('essential')
    newpkgs['packages'].append(pkg)

with open(get_data_dir() / "debian.yaml", "w") as f:
    yaml.dump(newpkgs, f, indent=2)


