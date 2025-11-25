import os
import ast
import re 

# Caminho da pasta com os arquivos de entrada
input_folder = "outputs_ballena" 
output_folder = "outputs_llms_ballena"
os.makedirs(output_folder, exist_ok=True)

# Prefixos
prefixos_padrao = ["qwen14b: ", "phi14b: ", "llama8b: "]
prefixos_duas = ["qwen14b: ", "llama8b: "]

# Regex para capturar listas aninhadas separadas corretamente
def extrair_listas(texto):
    matches = list(re.finditer(r'(\[\[.*?\]\])', texto))
    listas = [ast.literal_eval(m.group(1)) for m in matches]
    return listas

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Se já estiver formatado, ignore
    if any(content.startswith(prefix) for prefix in prefixos_padrao):
        print(f"[SKIP] {filename} já está formatado.")
        continue

    try:
        listas_extraidas = extrair_listas(content)

        if len(listas_extraidas) == 3:
            linhas_formatadas = [
                f"{prefixos_padrao[i]}{listas_extraidas[i]}" for i in range(3)
            ]
        elif len(listas_extraidas) == 2:
            linhas_formatadas = [
                f"{prefixos_duas[i]}{listas_extraidas[i]}" for i in range(2)
            ]
        elif len(listas_extraidas) == 1:
            linhas_formatadas = [
                f"{prefixos_duas[i]}{listas_extraidas[i]}" for i in range(1)
            ]
        else:
            print(f"[ERRO] {filename} - esperado 1, 2 ou 3 listas, mas nao encontrou.")
            continue

        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(linhas_formatadas))

        print(f"[OK] {filename} processado com sucesso.")

    except Exception as e:
        print(f"[ERRO] {filename}: {e}")