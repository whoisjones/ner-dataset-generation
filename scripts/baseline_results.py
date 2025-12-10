import os
import json
import pandas as pd

def main():
    results = []
    for result_dir in [
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gliner_multi-v2.1/evaluation", 
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gliner_multi-v2.1/evaluation_translated", 
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gliner-x-base/evaluation", 
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gliner-x-base/evaluation_translated", 
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gliner-x-large/evaluation", 
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gliner-x-large/evaluation_translated",
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gemma3-27b",
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/qwen3-32b",
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gpt5",
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/wikineural"
    ]:
        if "gliner" in result_dir:
            model_name = result_dir.split("/")[-2]
        else:
            model_name = result_dir.split("/")[-1]
        for dataset_dir in os.listdir(result_dir):
            for language_result_file in os.listdir(f"{result_dir}/{dataset_dir}"):
                language_code = language_result_file.split(".")[0]
                if not "binder" in result_dir:
                    with open(f"{result_dir}/{dataset_dir}/{language_result_file}", "r") as f:
                        result = json.load(f)

                    if 'gliner' in result_dir:
                        f1 = result['overall']['f_score']
                    else:
                        f1 = result['f1']
                    dataset_name = dataset_dir if "evaluation_translated" not in result_dir else dataset_dir + " (translated)"
                
                else:
                    if not "results.json" in language_result_file:
                        continue

                    if "all_results.json" in language_result_file:
                        continue

                    with open(f"{result_dir}/{dataset_dir}/{language_result_file}", "r") as f:
                        result = json.load(f)

                    f1 = result['test_f1']
                    model_name = "binder"
                    language_code = language_code.split("_")[0]
                    dataset_name = dataset_dir if "translated" not in dataset_dir else dataset_dir.split("_")[0] + " (translated)"

                results.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'language': language_code,
                    'f1': f1
                })
    
    df = pd.DataFrame(results)
    df = df.groupby(['model', 'dataset']).agg({'f1': 'mean'}).reset_index()
    df = df.pivot(index='model', columns='dataset', values='f1')
    df['avg'] = df.mean(axis=1)
    df = df.sort_values(by='avg', ascending=False)
    print(df.to_latex())

def gliner_string_evaluation():
    results = []
    for result_dir in [
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gliner_string_eval/gliner_multi-v2.1", 
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gliner_string_eval/gliner-x-base", 
        "/vol/tmp/goldejon/ner/paper_results/baseline_results/gliner_string_eval/gliner-x-large", 
    ]:
        model_name = result_dir.split("/")[-1]
        for dataset_dir in os.listdir(result_dir):
            for language_result_file in os.listdir(f"{result_dir}/{dataset_dir}"):
                language_code = language_result_file.split(".")[0]
                with open(f"{result_dir}/{dataset_dir}/{language_result_file}", "r") as f:
                    result = json.load(f)

                f1 = result['micro_average']['f1']

                results.append({
                    'model': model_name,
                    'dataset': dataset_dir,
                    'language': language_code,
                    'f1': f1
                })
    
    df = pd.DataFrame(results)
    df = df.groupby(['model', 'dataset']).agg({'f1': 'mean'}).reset_index()
    df = df.pivot(index='model', columns='dataset', values='f1')
    df['avg'] = df.mean(axis=1)
    df = df.sort_values(by='avg', ascending=False)
    print(df.to_latex())

if __name__ == "__main__":
    gliner_string_evaluation()