import requests

class get_client(object):
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


def format_language(languages, output_format):
    if len(languages) == 0:
        return ''
    s = sorted(languages, key= lambda l: l['score'], reverse=True)
    if output_format == 'iso':
        return s[0]['iso6391Name']
    else:
        return s[0]['name']

def format_sentiment(score, scale):
    if scale == 'binary':
        return 'negative' if score < 0.5 else 'positive'
    elif scale == 'ternary':
        return 'negative' if score < 0.33 else 'positive' if score > 0.66 else 'neutral'
    elif scale == '1to5':
        if score < 0.1:
            return 'highly negative'
        elif score < 0.33:
            return 'negative'
        elif score < 0.66:
            return 'neutral'
        elif score < 0.9:
            return 'positive'
        else:
            return 'highly positive'
    elif scale == 'continuous':
        return score
    else:
        raise ValueError("Invalid sentiment scale")