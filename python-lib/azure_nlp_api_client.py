# -*- coding: utf-8 -*-
import logging
import os
import requests

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


API_EXCEPTIONS = requests.RequestException

API_SUPPORT_BATCH = True
BATCH_RESULT_KEY = "documents"
BATCH_ERROR_KEY = "errors"
BATCH_INDEX_KEY = "id"
BATCH_ERROR_MESSAGE_KEY = "message"
BATCH_ERROR_TYPE_KEY = ""

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class get_client:
    def __init__(self, api_configuration_preset):
        if api_configuration_preset is None or api_configuration_preset == {}:
            raise ValueError(
                "No Azure credentials provided, please enter an API configuration preset"
            )
        self.api_key = str(api_configuration_preset.get("azure_api_key", ""))
        self.region = str(api_configuration_preset.get("azure_region", ""))
        self.endpoint = "https://{}.api.cognitive.microsoft.com".format(self.region)
        if self.api_key is None or self.api_key == "":
            self.api_key = os.environ["AZURE_TEXT_ANALYTICS_KEY"]
        if self.region is None or self.region == "":
            self.endpoint = os.environ["AZURE_TEXT_ANALYTICS_ENDPOINT"]
        self.headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key,
        }
        self.version = "v3.0-preview.1"
        self.base_url = "{}/text/analytics/{}/".format(self.endpoint, self.version)
        logging.info("Credentials loaded")

    def _post(self, service, data):
        response = requests.post(
            self.base_url + service, json=data, headers=self.headers
        )
        return response.json()

    def detect_language(self, data):
        return self._post("languages", data)

    def analyze_sentiment(self, data):
        return self._post("sentiment", data)

    def recognize_entities(self, data):
        return self._post("entities", data)

    def extract_keyphrases(self, data):
        return self._post("keyPhrases", data)
