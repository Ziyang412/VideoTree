import json
import re

# Load the JSON data from 1.json
with open('/qa/output/path ', 'r') as file:
    json_data = json.load(file)


# Function to extract prediction and confidence from response and update the item
def update_pred_confidence(data):
    prediction_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for uid, item in data.items():
        # print("`item` is", item)
        response = item.get("response", "")
        # print("response",response)
        item["confidence"] = 0
        if response == None:
            item["pred"] = 2
            item["confidence"] = 1
            continue
        prediction_match = re.search(r"prediction: ([A-E])", response, re.IGNORECASE)
        confidence_match = re.search(r"confidence: (\d+)", response, re.IGNORECASE)
        if prediction_match:
            # Update 'pred' with the numerical value of the prediction
            item["pred"] = prediction_map[prediction_match.group(1).upper()]
        if confidence_match:
            # Add a new 'confidence' field with the extracted confidence level
            item["confidence"] = int(confidence_match.group(1))

# Call the function to update the JSON data
update_pred_confidence(json_data["data"])
# update_pred_confidence(json_data)


# Save the updated JSON data into 2.json
with open('/final/results/output/json/path', 'w') as file:
    json.dump(json_data, file, indent=4)
