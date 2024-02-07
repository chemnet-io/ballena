import json
from collections import Counter

def aggregate_json_values(json_array):
    # Load the JSON data into a Python object if it's a string
    if isinstance(json_array, str):
        data = json.loads(json_array)
    else:
        data = json_array

    # Initialize a dictionary to hold counters for each field
    field_counters = {}

    # Populate field_counters with a Counter for each field
    for item in data:
        for field, value in item.items():
            if field not in field_counters:
                field_counters[field] = Counter()
            field_counters[field][value] += 1

    # Prepare the result dictionary
    result = {}

    # Determine the most common value or concatenate if there's no majority
    for field, counter in field_counters.items():
        # Extract values and their counts
        values, counts = zip(*counter.most_common())

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

# Example usage:
json_data = '''
[
    {"name": "John Doe", "location": "New York", "occupation": "Developer"},
    {"name": "Jane Doe", "location": "Los Angeles", "occupation": "Designer"}
]
'''

# Call the function with the JSON data
result_json = aggregate_json_values(json_data)

# Print the result
print("Resulting JSON data:")
print(result_json)
