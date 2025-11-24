import subprocess
import time

folds = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
tarefas = ['collectionType', 'collectionSite', 'bioActivity', 'collectionSpecie', 'name']
models = ['llama8b']

fold = '3'

for model in models:
    for estagio in ['1st', '2nd', '3rd', '4th']:
        for tarefa in tarefas:
        #for fold in folds:
            with open('nohups-ablation/' + model +  '_' + tarefa + '_' + estagio + '_' + fold +'.out', 'w') as outfile:
                subprocess.run(
                    ['nohup', 'python3', 'finetuning.py', '--model' , model , '--tarefa', tarefa, '--estagio', estagio, '--fold', fold],
                    stdout=outfile,
                    stderr=subprocess.STDOUT
                )
            time.sleep(60)
