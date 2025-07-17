import ast

tasks = ['bioActivity', 'collectionSite', 'collectionSpecie', 'collectionType', 'name']
evaluation_stages = ['1st', '2nd', '3rd', '4th']
model_types = ['pre-trained', 'finetuning']

# tasks = ['bioActivity']
# evaluation_stages = ['1st']
# model_types = ['pre-trained']

model_pred = {'model': [], 'pred_list': []}
for model_type in model_types:
    for task in tasks:
        for stage in evaluation_stages:
            for fold in range(10):
                with open(f"{model_type}/{task}_{stage}_{fold}", 'r', encoding='utf-8') as f:
                    for line in f:
                        split_line = line.split(': ')
                        model_pred['model'].append(split_line[0])
                        model_pred['pred_list'].append(ast.literal_eval(split_line[1]))

print(model_pred['pred_list'][0][3][1])