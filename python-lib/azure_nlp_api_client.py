# -*- coding: utf-8 -*-
"""Module with utility functions to call the Azure Cognitive Services Text Analytics API"""

import logging
import os
import requests
import json
from typing import Dict, List, Union, NamedTuple

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = requests.RequestException

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class AzureNLPAPIWrapper:
    def __init__(self, api_configuration_preset):
        if not api_configuration_preset:
            raise ValueError("No Azure credentials provided, please enter an API configuration preset")
        self.api_key = str(api_configuration_preset.get("azure_api_key", ""))
        self.region = str(api_configuration_preset.get("azure_region", ""))
        self.endpoint = "https://{}.api.cognitive.microsoft.com".format(self.region)
        if not self.api_key:
            self.api_key = os.environ["AZURE_TEXT_ANALYTICS_KEY"]
        if not self.region:
            self.endpoint = os.environ["AZURE_TEXT_ANALYTICS_ENDPOINT"]
        self.headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key,
        }
        self.version = "v3.0"
        self.base_url = "{}/text/analytics/{}/".format(self.endpoint, self.version)
        logging.info("Credentials loaded")

    def _post(self, service, data):
        response = requests.post(self.base_url + service, json=data, headers=self.headers)
        return response.json()

    def detect_language(self, data):
        return self._post("languages", data)

    def analyze_sentiment(self, data):
        return self._post("sentiment", data)

    def recognize_entities_general(self, data):
        return self._post("entities/recognition/general", data)

    def recognize_entities_pii(self, data):
        return self._post("entities/recognition/pii", data)

    def extract_keyphrases(self, data):
        return self._post("keyPhrases", data)


def batch_api_response_parser(batch: List[Dict], response: Union[Dict, List], api_column_names: NamedTuple) -> Dict:
    """
    Function to parse API results in the batch case. Needed for api_parallelizer.api_call_batch
    when APIs result need specific parsing logic (every API may be different).
    """
    results = response.get("documents", [])
    errors = response.get("errors", [])
    for i in range(len(batch)):
        for k in api_column_names:
            batch[i][k] = ""
        result = [r for r in results if str(r.get("id", "")) == str(i)]
        error = [r for r in errors if str(r.get("id", "")) == str(i)]
        if "error" in response and "documents" not in response:
            # case when the whole batch fails with a single error key
            error = [response]
        if len(result) != 0:
            # result must be json serializable
            batch[i][api_column_names.response] = json.dumps(result[0])
        if len(error) != 0:
            raw_error = error[0]
            logging.warning(str(raw_error))
            error_dict = raw_error.get("error", {})
            if "innererror" in error_dict:
                error_dict = error_dict.get("innererror", {})
            batch[i][api_column_names.error_message] = error_dict.get("message", str(raw_error))
            batch[i][api_column_names.error_type] = error_dict.get("code", "Undefined API error")
            batch[i][api_column_names.error_raw] = str(raw_error)
    return batch
