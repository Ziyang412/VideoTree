import json
import re


with open('relevance/scoring/output/file', 'r') as file:
    json_data = json.load(file)

def update_pred_confidence(data):
    prediction_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for uid, item in data.items():
        response = item.get("response", "")
        item["confidence"] = 0  # Initialize confidence to zero
        item["relevance"] = []  # Initialize relevance to an empty list
        if response is None:
            item["pred"] = 2
            item["confidence"] = 1
            continue
        # Extract prediction
        prediction_match = re.search(r"prediction: ([A-E])", response, re.IGNORECASE)
        if prediction_match:
            item["pred"] = prediction_map[prediction_match.group(1).upper()]
        # Extract confidence
        confidence_match = re.search(r"confidence: (\d+)", response, re.IGNORECASE)
        if confidence_match:
            item["confidence"] = int(confidence_match.group(1))
        # Extract relevance
        relevance_match = re.search(r"frame relevance: \[([0-9, ]+)\]", response)
        if relevance_match:
            # Convert the matched string to a list of integers
            item["relevance"] = list(map(int, relevance_match.group(1).split(',')))
            # Ensure relevance list has exactly four elements, appending 3 if necessary
            while len(item["relevance"]) < 8:
                item["relevance"].append(3)

# Call the function to update the JSON data
update_pred_confidence(json_data["data"])
# update_pred_confidence(json_data)


# Save the updated JSON data into 2.json
with open('/relevance/output/json/file', 'w') as file:
    json.dump(json_data, file, indent=4)

