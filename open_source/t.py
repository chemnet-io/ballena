from collections import defaultdict
import re

def parse_line(line):
    pattern = r"Tarefa: (.*?) \| Estagio: (.*?) \| Fold: (\d+) \| (Hits@\d+): ([0-9.]+)"
    match = re.match(pattern, line.strip())
    if match:
        tarefa, estagio, fold, metrica, valor = match.groups()
        return tarefa, estagio, int(fold), metrica, float(valor)
    return None

def process_file(file_path):
    resultados = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                tarefa, estagio, _, metrica, valor = parsed
                resultados[(tarefa, estagio, metrica)].append(valor)

    # Gera resumo
    resumo = []
    for (tarefa, estagio, metrica), valores in sorted(resultados.items()):
        media = sum(valores) / len(valores)
        resumo.append((tarefa, estagio, metrica, len(valores), media))

    return resumo

def imprimir_resumo(resumo):
    print(f"{'Tarefa':<20} {'Estágio':<10} {'Métrica':<10} {'Folds':<6} {'Média':<10}")
    print("-" * 65)
    for tarefa, estagio, metrica, num_folds, media in resumo:
        print(f"{tarefa:<20} {estagio:<10} {metrica:<10} {num_folds:<6} {media:<10.4f}")

# Exemplo de uso
if __name__ == "__main__":
    caminho_arquivo = "resultados-FT-phi14b.txt"  
    resumo = process_file(caminho_arquivo)
    imprimir_resumo(resumo)
