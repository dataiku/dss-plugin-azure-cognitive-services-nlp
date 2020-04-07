import logging
import time
import json
import dataiku
from dataiku.customrecipe import *
from dku_azure_cs import *
from common import *

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[azure-cognitive-services plugin] %(levelname)s - %(message)s')

connection_info = get_recipe_config().get('connection_info')
text_column = get_recipe_config().get('text_column')
language = get_recipe_config().get('language', '')
sentiment_scale = get_recipe_config().get('sentiment_scale', 'ternary')
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_dataset_name = get_input_names_for_role('input_dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]
predicted_sentiment_column = generate_unique('predicted_sentiment', input_columns_names)
predicted_probability_column = generate_unique('predicted_probability', input_columns_names)

output_dataset_name = get_output_names_for_role('output_dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

if text_column is None or len(text_column) == 0:
    raise ValueError("You must specify the input text column")
if text_column not in input_columns_names:
    raise ValueError("Column '{}' is not present in the input dataset".format(text_column))

#==============================================================================
# RUN
#==============================================================================

input_df = input_dataset.get_dataframe()

@with_original_indices
def detect_sentiment(text_list):
    client = get_client(connection_info)
    documents = [{"id": i, "text": text, "language": language} for (i, text) in enumerate(text_list)]
    logging.info("request: %d items / %d characters" % (len(text_list), sum([len(t) for t in text_list])))
    start = time.time()
    response = client.sentiment({"documents": documents})
    logging.info("request took %.3fs" % (time.time() - start))
    return response


for batch in run_by_batch(detect_sentiment, input_df, text_column, batch_size=BATCH_SIZE, parallelism=PARALLELISM):
    response, original_indices = batch
    if response.get('errors'):
        logging.error(response.get('errors'))
    for i, raw_result in enumerate(response.get("documents")):
        j = original_indices[i]
        output = format_sentiment_results(raw_result, sentiment_scale)
        input_df.set_value(j, predicted_sentiment_column, output['predicted_sentiment'])
        if should_output_raw_results:
            input_df.set_value(j, 'raw_results', json.dumps(output['raw_results']))

output_dataset.write_with_schema(input_df)