import json

with open("small-117M-k40.test.jsonl", "r") as input_file,open("output.txt", "w", encoding="utf-8") as output_file:
    for line in input_file:
        data = json.loads(line)
        text = data["text"]
        output_file.write(text + "\n")
