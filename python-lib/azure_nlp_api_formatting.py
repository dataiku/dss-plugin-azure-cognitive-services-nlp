# -*- coding: utf-8 -*-
import logging
from typing import AnyStr, Dict, List
from enum import Enum

import pandas as pd

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    ErrorHandlingEnum,
    build_unique_column_names,
    generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
)


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


class EntityTypeEnum(Enum):
    DateTime = "Date and Time entities"
    Email = "Email"
    Event = "Event"
    IPAddress = "IP Address"
    Location = "Location"
    Organization = "Organization"
    Person = "Person"
    PersonType = "Job type or role"
    PhoneNumber = "Phone number"
    Product = "Product"
    Quantity = "Quantity"
    Skill = "Skill"
    URL = "URL"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GenericAPIFormatter:
    """
    Geric Formatter class for API responses:
    - initialize with generic parameters
    - compute generic column descriptions
    - apply format_row to dataframe
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.api_column_names = build_unique_column_names(input_df, column_prefix)
        self.column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k] for k, v in self.api_column_names._asdict().items()
        }

    def format_row(self, row: Dict) -> Dict:
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names, self.error_handling)
        logging.info("Formatting API results: Done.")
        return df


class LanguageDetectionAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Language Detection API responses:
    - make sure response is valid JSON
    - extract language code from response
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "lang_detect_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.language_name_column = generate_unique("language_name", input_df.keys(), self.column_prefix)
        self.language_code_column = generate_unique("language_code", input_df.keys(), self.column_prefix)
        self.language_score_column = generate_unique("language_score", input_df.keys(), self.column_prefix)
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[self.language_name_column] = "Language name detected by the API"
        self.column_description_dict[self.language_code_column] = "Language code in ISO 639 format"
        self.column_description_dict[self.language_score_column] = "Confidence score of the API from 0 to 1"

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        language = response.get("detectedLanguage", {})
        row[self.language_name_column] = language.get("name", "")
        row[self.language_code_column] = language.get("iso6391Name", "")
        row[self.language_score_column] = None
        if language.get("confidenceScore") is not None:
            row[self.language_score_column] = float(language.get("confidenceScore"))
        return row


class SentimentAnalysisAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Sentiment Analysis API responses:
    - make sure response is valid JSON
    - extract sentiment scores from response
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "sentiment_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.sentiment_prediction_column = generate_unique("prediction", input_df.keys(), column_prefix)
        self.sentiment_score_column_dict = {
            p: generate_unique("score_" + p.lower(), input_df.keys(), column_prefix)
            for p in ["positive", "neutral", "negative"]
        }
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[
            self.sentiment_prediction_column
        ] = "Sentiment prediction from the API (positive/neutral/negative)"
        for prediction, column_name in self.sentiment_score_column_dict.items():
            self.column_description_dict[column_name] = "Confidence score in the {} prediction from 0 to 1".format(
                prediction.upper()
            )

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        row[self.sentiment_prediction_column] = response.get("sentiment", "")
        sentiment_score = response.get("confidenceScores", {})
        for prediction, column_name in self.sentiment_score_column_dict.items():
            row[column_name] = None
            score = sentiment_score.get(prediction)
            if score is not None:
                row[column_name] = round(score, 3)
        return row


class NamedEntityRecognitionAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Named Entity Recognition API responses:
    - make sure response is valid JSON
    - expand results to multiple columns (one by entity type)
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        entity_types: List,
        minimum_score: float,
        column_prefix: AnyStr = "entity_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.entity_types = entity_types
        self.minimum_score = float(minimum_score)
        self._compute_column_description()

    def _compute_column_description(self):
        for n, m in EntityTypeEnum.__members__.items():
            entity_type_column = generate_unique("entity_type_" + n.lower(), self.input_df.keys(), self.column_prefix)
            self.column_description_dict[entity_type_column] = "List of '{}' entities recognized by the API".format(
                str(m.value)
            )

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        entities = response.get("entities", [])
        selected_entity_types = sorted([e.name for e in self.entity_types])
        discarded_entities = [
            e
            for e in entities
            if float(e.get("confidenceScore", 0)) < self.minimum_score
            and e.get("category", "") in selected_entity_types
        ]
        if len(discarded_entities) != 0:
            logging.info("Discarding {} entities below the minimum score threshold".format(len(discarded_entities)))
        for n in selected_entity_types:
            entity_type_column = generate_unique("entity_type_" + n.lower(), row.keys(), self.column_prefix)
            row[entity_type_column] = [
                e.get("text")
                for e in entities
                if e.get("category", "") == n and float(e.get("confidenceScore", 0)) >= self.minimum_score
            ]
            if len(row[entity_type_column]) == 0:
                row[entity_type_column] = ""
        return row


class PIIExtractionAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for PII Extraction API responses:
    - make sure response is valid JSON
    - extract list of PII data
    - output a redacted version of the input column without PII
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        minimum_score: float,
        column_prefix: AnyStr = "pii_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.text_column = str(text_column)
        self.minimum_score = float(minimum_score)
        self.pii_column_text = generate_unique("text_list", self.input_df.keys(), self.column_prefix)
        self.pii_column_raw = generate_unique("raw_list", self.input_df.keys(), self.column_prefix)
        self.pii_column_redacted = generate_unique("redacted", self.input_df.keys(), self.column_prefix)
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[self.pii_column_text] = "List of PII in text format"
        self.column_description_dict[self.pii_column_raw] = "List of PII in JSON format"
        self.column_description_dict[self.pii_column_redacted] = "Redacted version of the input column without PII"

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        entities = [
            e
            for e in response.get("entities", [])
            if e.get("text", "") != "" and e.get("category", "") not in {"Organization", "DateTime", "Quantity"}
        ]
        entities_filtered = [e for e in entities if float(e.get("confidenceScore", 0)) >= self.minimum_score]
        discarded_entities = [e for e in entities if float(e.get("confidenceScore", 0)) < self.minimum_score]
        if len(discarded_entities) != 0:
            logging.info("Discarding {} entities below the minimum score threshold".format(len(discarded_entities)))
        text_to_redact = str(row.get(self.text_column, ""))
        row[self.pii_column_redacted] = ""
        row[self.pii_column_text] = ""
        row[self.pii_column_raw] = ""
        if len(entities_filtered) != 0:
            row[self.pii_column_raw] = entities_filtered
            row[self.pii_column_text] = [e.get("text") for e in entities_filtered]
            entities_reversed = sorted(entities_filtered, key=lambda x: int(x.get("offset", -1)), reverse=True)
            for e in entities_reversed:
                start = int(e.get("offset"))
                end = start + int(e.get("length"))
                text_to_redact = text_to_redact[:start] + "" + text_to_redact[end:]
            row[self.pii_column_redacted] = text_to_redact
        return row


class KeyPhraseExtractionAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Key Phrase Extraction API responses:
    - make sure response is valid JSON
    - extract a given number of key phrases
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        num_key_phrases: int,
        column_prefix: AnyStr = "keyphrase_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.num_key_phrases = num_key_phrases
        self._compute_column_description()

    def _compute_column_description(self):
        for n in range(self.num_key_phrases):
            keyphrase_column = generate_unique("keyphrase_" + str(n + 1), self.input_df.keys(), self.column_prefix,)
            self.column_description_dict[keyphrase_column] = "Keyphrase {} extracted by the API".format(str(n + 1))

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        key_phrases = response.get("keyPhrases", [])
        for n in range(self.num_key_phrases):
            keyphrase_column = generate_unique("keyphrase_" + str(n + 1), row.keys(), self.column_prefix)
            if len(key_phrases) > n:
                row[keyphrase_column] = key_phrases[n]
            else:
                row[keyphrase_column] = ""
        return row
