PLUGIN_VERSION=2.0.0
PLUGIN_ID=azure-cognitive-services-nlp

plugin:
	cat plugin.json|json_pp > /dev/null
	rm -rf dist
	mkdir dist
	zip --exclude "*.pyc" -r dist/dss-plugin-${PLUGIN_ID}-${PLUGIN_VERSION}.zip plugin.json python-lib custom-recipes parameter-sets code-env
