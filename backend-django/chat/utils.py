import pandas as pd
import json
import ast
from typing import List


def get_actors(item: int, credits: pd.DataFrame) -> List[str]:
    d = ast.literal_eval(credits["cast"][item - 1])
    json_str = json.dumps(d, indent=4)
    d = pd.read_json(json_str)

    # Error handling check
    if len(d) > 0:
        actors = d["name"][:3].to_list()
    else:
        actors = []

    return actors
