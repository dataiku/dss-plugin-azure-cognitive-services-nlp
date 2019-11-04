# Microsoft Azure Cognitive Services - NLP


## Plugin information

This Dataiku DSS plugin provides several tools to interact with the Natural Language API from [Microsoft Azure Cognitive Services API](https://azure.microsoft.com/en-us/services/cognitive-services/).

The Text Analytics API is a suite of text analytics web services built with best-in-class Microsoft machine learning algorithms.
The API can be used to analyze unstructured text for tasks such as sentiment analysis, key phrase extraction and language detection.
No training data is needed to use this API; just bring your text data. This API uses advanced natural language processing techniques to
deliver best in class predictions.
[Read the documentation](https://westus.dev.cognitive.microsoft.com/docs/services/TextAnalytics.V2.0/operations/56f30ceeeda5650db055a3c7)
for more information.


## Using the Plugin

### Prerequisites
In order to use the Plugin, you will need:

* an Azure account
* proper credentials (access tokens) to interact with the service:
	1. Sign in to [Azure portal](https://portal.azure.com/).
	2. In the left navigation pane, select **All services**.
	3. In Filter, type Cognitive Services. Add the **Text Analytics** service
	4. Select a plan
* make sure you know in **which Azure region the services are valid**, the Plugin will need this information to get authenticated

### Plugin components
The Plugin has the following components:

* [Language Detection](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/how-tos/text-analytics-how-to-language-detection):
evaluates text input and for each document and returns language identifiers with a score indicating the strength of the analysis.
Text Analytics recognizes up to 120 languages.
* [Sentiment Analysis](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/how-tos/text-analytics-how-to-sentiment-analysis):
evaluates text input and returns a sentiment score for each document, ranging from 0 (negative) to 1 (positive). This capability
is useful for detecting positive and negative sentiment in social media, customer reviews, and discussion forums.
Content is provided by you; models and training data are provided by the service.
* [Key Phrases Extraction](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/how-tos/text-analytics-how-to-keyword-extraction):
evaluates unstructured text, and for each JSON document, returns a list of key phrases. This capability is useful if you need to quickly
identify the main points in a collection of documents. For example, given input text "The food was delicious and there were wonderful staff",
the service returns the main talking points: "food" and "wonderful staff".
* [Named Entity Recognition](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/how-tos/text-analytics-how-to-entity-linking):
takes unstructured text, and for each JSON document, returns a list of disambiguated entities with links to more information on the web (Wikipedia and Bing).
