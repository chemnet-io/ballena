import json
from collections import Counter
import re  # Import the regular expressions library

def aggregate_json_values(json_array):
    # Load the JSON data into a Python object if it's a string
    if isinstance(json_array, str):
        data = json.loads(json_array)
    else:
        data = json_array

    # Initialize a dictionary to hold counters for each field
    field_counters = {}

    # Populate field_counters with a Counter for each field, ignoring empty or non-alphanumeric values
    for item in data:
        for field, value in item.items():
            # Use regular expression to check if the value is empty or does not contain alphanumeric characters
            if not re.search('[a-zA-Z0-9]', value):
                continue  # Skip if value is empty or has no alphanumeric characters
            if field not in field_counters:
                field_counters[field] = Counter()
            field_counters[field][value] += 1

    # Prepare the result dictionary
    result = {}

    # Determine the most common value or concatenate if there's no majority
    for field, counter in field_counters.items():
        # Extract values and their counts
        values, counts = zip(*counter.most_common()) if counter else ([], [])

        # Check if there's a single most common value
        if len(counter) == 1 or counter.most_common(1)[0][1] > 1:
            result[field], _ = counter.most_common(1)[0]
        else:
            # Concatenate all values if no clear majority and ensure unique values
            unique_values = sorted(set(values))
            result[field] = ",".join(unique_values)

    # Convert the result dictionary into a JSON string
    result_json_data = json.dumps(result, ensure_ascii=False)

    return result_json_data
