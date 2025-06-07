import json
from string import Template

instruction = Template(
    """Imagine you are an assistant who receives a text query and five text descriptions. You need to decide which text description is cloest to the text query. 
    
    Requirments of your job:
    1. Rank the text descriptions against the text query, i.e., decide which text description is closest to the text query.
    2. Just give me a single number of which desciption is closest to the text query. For example, give me number 2 if the second description is closet.
    3. Do not add any comments or notes like 'Here is my thought.' in the output result. Just give me one single number.
    
    Here are some examples: ${examples}

    Here is the text query: ${text_query}
    Here are the five text descriptions:
    text description 1: ${text_description_1}
    text description 2: ${text_description_2}
    text description 3: ${text_description_3}
    text description 4: ${text_description_4}
    text description 5: ${text_description_5}
    
    Please go ahead giving me the ranking number.
    """
)


examples = [
    {
        "text query": "Brown Bear Ursus Arctos Distant growls, Distant gulls at times, Running water in background.",
        "text description 1": "Lion runs, river is flowing.",
        "text description 2": "Brown bear shouts at distance, gulls show up at times, water is flowing in the background.",
        "text description 3": "Bear shouts, people are talking.",
        "text description 4": "Car horn buzzles.",
        "text description 5": "Birds sing at times, water is flushing.",
        "your ranking result": "2"
    },
    {
        "text query": "Thunder - Thunder without rain. Some water droplets. Cicadas in background.",
        "text description 1": "Thunder and store, very heavy rain.",
        "text description 2": "Birds sing, water is flowing.",
        "text description 3": "Lion shouts, people are talking.",
        "text description 4": "Car horn buzzles.",
        "text description 5": "Thunder in the sky. Water drops. Cicadas sound.",
        "your ranking result": "5"
    }
]

examples = f"""{json.dumps(examples, ensure_ascii=False)}"""
