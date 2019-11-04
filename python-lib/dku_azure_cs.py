import requests

class NLPClient(object):
    def __init__(self, connection_info):
        if connection_info is None:
            raise ValueError("No Azure credentials provided")
        if connection_info.get('api_key') is None:
            raise ValueError("Bad credentials: API Key not provided")
        if connection_info.get('azure_location') is None:
            raise ValueError("Bad credentials: Azure location not provided")
        self.headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': connection_info.get('api_key')
        }
        self.base_url = "https://{}.api.cognitive.microsoft.com/text/analytics/v2.0/".format(connection_info.get('azure_location'))

    def sentiment(self, data):
        return self._post("sentiment", data)

    def keyphrases(self, data):
        return self._post("keyPhrases", data)

    def entities(self, data):
        return self._post("entities", data)

    def languages(self, data):
        return self._post("languages", data)

    def _post(self, service, data):
        r = requests.post(self.base_url+service, json=data, headers=self.headers)
        if r.text == '':
            raise Exception("Empty answer from Azure Cognitive Services (HTTP code: " + str(r.status_code) + ")")
        return r.json()
