import jsonlines

with jsonlines.open('./Dataset/small-117M-k40.train.jsonl') as reader:
    with open('./New-Dataset/small-117M-k40.train.txt', 'w', encoding="utf-8") as writer:
        for obj in reader:
            writer.write(obj['text'] + '\n')
