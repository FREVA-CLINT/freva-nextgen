.PHONY: all release docs
all: install

install: prepare
	python3 -m pip install -e ./freva-rest[dev] -e ./freva-client -e ./freva-data-portal-worker[full]

prepare:
	python3 -m pip install cryptography tox
	mkdir -p dev-env/certs
	python3 dev-env/config/dev-utils.py gen-certs

lint: types
	tox -e lint

types:
	tox -e types

docs:
	tox -e docs

release:
	tox -e release
