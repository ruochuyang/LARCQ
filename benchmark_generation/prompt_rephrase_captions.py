import json
from string import Template

instruction = Template(
    """Imagine you are an assistant which has a input box of text queries.
    A user uploads a text query and you accept this query as input.

    Requirments of your job:
    1. Try to rephrase the input query while maintain all information of the input query.
    2. If the input query is meangingless, you can just keep it unchanged.
    3. Your rephrased query should be always relevant to the original query. Do not add any external knowledge or your imagination.
    4. Do not add any description text like 'Here is the rephrased query based on the given query:' in the output query. Just output the rephrased query itself.
    5. Do not add any comments or notes like 'Here I made some slight modifications to the input query:' in the output query. Just output the rephrased query itself.
    
    Here are some examples: ${examples}

    Here is the input text query: "${input_query}"
    Please go ahead rephrasing the input query.
    """
)


examples = [
    {
        "input query": "Brown Bear Ursus Arctos Distant growls, Distant gulls at times, Running water in background",
        "rephrased query": "Brown bear shouts at distance, gulls show up at times, water is flowing in the background"
    },
    {
        "input query": "Tropical Forest Wet Evergreen NIGHT In cocoa plantation with frogs",
        "rephrased query": "At night, there are frogs in cocoa plantation inside the wet and evergreen tropical forest"
    },
    {
        "input query": "Romania 2 Trolleybus pneumatic doors open",
        "rephrased query": "Pneumatic doors of the trolleybus named Romania 2 open"
    }
]

examples = f"""{json.dumps(examples, ensure_ascii=False)}"""
