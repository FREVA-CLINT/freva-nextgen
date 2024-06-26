.PHONY: install prepare lint docs
all: install

install: prepare
	python3 -m pip -e install freva-client freva-rest
	flit install --deps=all

prepare:
	python3 -m pip install cryptography tox
	mkdir -p dev-env/certs
	python3 dev-env/keys.py

lint:
	tox -e lint types

docs: tox -e docs

release: tox -e release
