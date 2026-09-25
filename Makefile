WHL_BUILD_DIR :=package

# default rule
default: whl

.PHONY: linter
linter:
	bash .dev_scripts/linter.sh

.PHONY: test
test:
	bash .dev_scripts/citest.sh

.PHONY: whl
whl:
	python setup.py sdist bdist_wheel

.PHONY: clean
clean:
	rm -rf  $(WHL_BUILD_DIR)
