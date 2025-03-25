import json


def return_data():
    data = []
    # max_len_text = ""
    # max_ = 0
    with open("../../Datasets/fin_corpus.jsonl") as f:
        for line in f:
            data.append(json.loads(line))
            # if(max_<len(data[len(data)-1]['text'])):
            #     max_ = len(data[len(data)-1]['text'])
            #     max_len_text = data[len(data)-1]['text']
            # print(data[0]['text'])
            # break
    # print(max_len_text)
    return data
if __name__ == "__main__":
    return_data()
        