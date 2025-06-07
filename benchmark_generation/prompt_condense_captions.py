import json
from string import Template

instruction = Template(
    """Imagine you are an assistant which accepts text captions as input.

    Requirments of your job:
    1. Condense the input caption.
    2. Your condensed caption should be always relevant to the input caption. Do not add any external knowledge or your imagination.
    4. Do not add any description text like 'Here is the condensed caption based on the given caption'. Just output the condensed caption itself.
    5. Do not add any comments or notes like 'Here I made some slight modifications to the input caption'. Just output the condensed caption itself.
    
    Here are some examples: ${examples}

    Here is the input caption: "${input_caption}"
    Please go ahead rephrasing the input caption.
    """
)


examples = [
    {
        "input caption": "Brown Bear Ursus Arctos Distant growls, Distant gulls at times, Running water in background",
        "condensed caption": "Bear growls with gulls and water in background"
    },
    {
        "input caption": "Tropical Forest Wet Evergreen NIGHT In cocoa plantation with frogs",
        "condensed caption": "Frogs in tropical forest"
    },
    {
        "input caption": "Romania 2 Trolleybus pneumatic doors open",
        "condensed caption": "Trolleybus doors open"
    }
]

examples = f"""{json.dumps(examples, ensure_ascii=False)}"""
