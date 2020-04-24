# -*- coding: utf-8 -*-
import logging
import os
from typing import AnyStr, Dict
from enum import Enum

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from plugin_io_utils import (
    generate_unique, safe_json_loads, ErrorHandlingEnum, OutputFormatEnum)


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (AzureError, TypeError, ValueError)

API_SUPPORT_BATCH = True
BATCH_RESULT_KEY = ""
BATCH_ERROR_KEY = ""
BATCH_INDEX_KEY = ""
BATCH_ERROR_MESSAGE_KEY = ""
BATCH_ERROR_TYPE_KEY = ""

APPLY_AXIS = 1  # columns


class EntityTypesEnum(Enum):
    FOO = "bar"


# ==============================================================================
# FUNCTION DEFINITION
# ==============================================================================


def get_client(api_configuration_preset):
    api_key = api_configuration_preset.get("azure_api_key", "")
    if str(api_key) == "":
        api_key = os.environ["AZURE_TEXT_ANALYTICS_KEY"]
    credential = AzureKeyCredential(api_key)
    region = api_configuration_preset.get("azure_region", "")
    endpoint = "https:/{}.api.cognitive.microsoft.com/".format(region)
    if str(region) == "":
        endpoint = os.environ["AZURE_TEXT_ANALYTICS_ENDPOINT"]
    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint, credential=credential)
    logging.info("Credentials loaded")
    return text_analytics_client


def format_language_detection(
    row: Dict,
    response_column: AnyStr,
    column_prefix: AnyStr = "lang_detect_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    language_column = generate_unique(
        "language_code", row.keys(), column_prefix)
    row[language_column] = ''
    languages = response.get("Languages", [])
    if len(languages) != 0:
        row[language_column] = languages[0].get("LanguageCode", "")
    return row


def format_key_phrase_extraction(
    row: Dict,
    response_column: AnyStr,
    output_format: OutputFormatEnum = OutputFormatEnum.MULTIPLE_COLUMNS,
    num_key_phrases: int = 3,
    column_prefix: AnyStr = "keyphrase_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    if output_format == OutputFormatEnum.SINGLE_COLUMN:
        key_phrase_column = generate_unique(
            "keyphrase_list", row.keys(), column_prefix)
        row[key_phrase_column] = response.get("KeyPhrases", "")
    else:
        key_phrases = sorted(
            response.get("KeyPhrases", []), key=lambda x: x.get("Score"),
            reverse=True)
        for n in range(num_key_phrases):
            keyphrase_column = generate_unique(
                "keyphrase_" + str(n), row.keys(), column_prefix)
            score_column = generate_unique(
                "keyphrase_" + str(n) + "_score", row.keys(), column_prefix)
            if len(key_phrases) > n:
                row[keyphrase_column] = key_phrases[n].get("Text", "")
                row[score_column] = key_phrases[n].get("Score")
            else:
                row[keyphrase_column] = ''
                row[score_column] = None
    return row


def format_named_entity_recognition(
    row: Dict,
    response_column: AnyStr,
    output_format: OutputFormatEnum = OutputFormatEnum.MULTIPLE_COLUMNS,
    column_prefix: AnyStr = "ner_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    if output_format == OutputFormatEnum.SINGLE_COLUMN:
        entity_column = generate_unique(
            "entities", row.keys(), column_prefix)
        row[entity_column] = response.get("Entities", "")
    else:
        entities = response.get("Entities", [])
        for entity_enum in EntityTypesEnum:
            entity_type_column = generate_unique(
                "entity_type_" + str(entity_enum.value).lower(),
                row.keys(), column_prefix)
            row[entity_type_column] = [
                e.get("Text") for e in entities
                if e.get("Type", "") == entity_enum.name]
            if len(row[entity_type_column]) == 0:
                row[entity_type_column] = ''
    return row


def format_sentiment_analysis(
    row: Dict,
    response_column: AnyStr,
    column_prefix: AnyStr = "sentiment_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    sentiment_column = generate_unique("sentiment", row.keys(), column_prefix)
    row[sentiment_column] = response.get("Sentiment", '')
    return row
