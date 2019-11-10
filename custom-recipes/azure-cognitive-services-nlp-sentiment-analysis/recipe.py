import logging
import dataiku
from dataiku.customrecipe import *
from dku_azure_cs import *

BATCH_SIZE = 20

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[azure-cognitive-services plugin] %(levelname)s - %(message)s')

connection_info = get_recipe_config().get('connection_info')
input_text_column = get_recipe_config().get('text-column')
language_column = get_recipe_config().get('language-column', '')
sentiment_scale = get_recipe_config().get('sentiment-scale', 'ternary')
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_dataset_name = get_input_names_for_role('input-dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]

output_dataset_name = get_output_names_for_role('output-dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

client = get_client(connection_info)

#==============================================================================
# RUN
#==============================================================================

output_schema = input_schema
output_column_name = generate_unique('predicted_sentiment', input_columns_names)
output_schema.append({'name': output_column_name, 'type': 'string'})
if should_output_raw_results:
    output_schema.append({"name": "raw_results", "type": "string"})
output_dataset.write_schema(output_schema)

with output_dataset.get_writer() as writer:
    for batch in input_dataset.iter_dataframes(chunksize=BATCH_SIZE):
        batch = batch.reset_index()
        documents = []
        indices_in_documents = [None] * len(batch)
        for i, row in batch.iterrows():
            row = row.fillna('')
            if row.get(input_text_column) is None or row.get(input_text_column) == '':
                continue
            doc = {"id": i, "text": row.get(input_text_column)}
            if len(language_column) > 0:
                doc["language"] = row[language_column]
            indices_in_documents[i] = len(documents)
            documents.append(doc)

        response = client.sentiment({"documents": documents})

        if response.get('errors'):
            logging.error(response.get('errors'))

        for i, row in batch.iterrows():
            row = row.fillna('')
            doc_index = indices_in_documents[i]
            if response.get("documents") and doc_index is not None:
                score = response.get("documents")[doc_index]["score"] # float from 0 (very negative) to 1 (very positive)
                row[output_column_name] = format_sentiment(score, sentiment_scale)
            writer.write_row_dict(row)
