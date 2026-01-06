import glob
import json
import pandas as pd
import os

def baselines():
    results = []
    for result_dir in ["/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gliner_multi-v2.1/evaluation", 
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gliner-x-base/evaluation", 
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gliner-x-large/evaluation", 
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gemma3-27b",
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/qwen3-32b",
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/gpt5",
                       "/vol/fob-vol7/mi18/goldejon/multilingual-ner/results/wikineural"]:
        model_name = result_dir.split("/")[-2] if "gliner" in result_dir else result_dir.split("/")[-1]
        for dataset_dir in os.listdir(result_dir):
            for language_result_file in os.listdir(f"{result_dir}/{dataset_dir}"):
                language_code = language_result_file.split(".")[0]
                with open(f"{result_dir}/{dataset_dir}/{language_result_file}", "r") as f:
                    result = json.load(f)

                if 'f1' in result:
                    f1 = result['f1']
                else:
                    f1 = result['overall']['f_score']

                results.append({
                    'model': model_name if "evaluation_translated" not in result_dir else model_name + " (translated)",
                    'dataset': dataset_dir,
                    'language': language_code,
                    'f1': f1
                }) 
    
    df = pd.DataFrame(results)
    return df

def main_results():
    rows = []
    for result_dir in [
        "/vol/tmp/goldejon/ner/paper_results/first_experiment/bi_rembert_finerweb",
        "/vol/tmp/goldejon/ner/paper_results/first_experiment/bi_mmbert_finerweb",
        "/vol/tmp/goldejon/ner/paper_results/first_experiment/ce_rembert_finerweb",
        "/vol/tmp/goldejon/ner/paper_results/first_experiment/ce_mmbert_finerweb",
        "/vol/tmp/goldejon/ner/paper_results/test_runs/bi_mmbert_finerweb_1M",
        "/vol/tmp/goldejon/ner/paper_results/test_runs/ce_mmbert_finerweb_1M",
    ]:
        for evaluation_dir in glob.glob(result_dir + "/*"):
            for evaluation_file in glob.glob(evaluation_dir + "/*.json"):
                with open(evaluation_file, "r") as f:
                    data = json.load(f)
                rows.append({
                    "model": result_dir.split("/")[-1],
                    "dataset": evaluation_dir.split("/")[-1],
                    "language": evaluation_file.split("/")[-1].split("_")[0],
                    "threshold": evaluation_file.split("/")[-1].split("_")[1].split(".j")[0],
                    "f1": data["test_metrics"]["micro"]["f1"],
                })

    df = pd.DataFrame(rows)
    df["threshold"] = (
        df["threshold"]
        .astype(float)
    )
    df = df[~df['dataset'].str.contains('translated')]
    df = df[~df['model'].str.contains('translated')]
    df_agg = df.copy(deep=True)
    df_agg['architecture'] = df['model'].str.extract(r'^(ce|bi)_')
    df_agg['backbone'] = df['model'].str.extract(r'_(rembert|mdeberta|xlmr|mmbert|mt5)_')
    df_agg['train_dataset'] = df['model'].str.extract(r'_(finerweb_1M|euroglinerx|pilener|finerweb)')
    df_lang_f1 = df_agg.groupby(["architecture", "threshold"]).agg({"f1": "mean"}).reset_index()
    df_dataset_f1 = df_agg.groupby(["architecture", 'backbone', 'train_dataset', "threshold", "dataset"]).agg({"f1": "mean"}).reset_index()
    df_dataset_f1 = df_dataset_f1.pivot(index=["architecture", 'backbone', 'train_dataset', "threshold"], columns="dataset", values="f1")
    df_dataset_f1["average"] = df_dataset_f1.mean(axis=1)
    df_dataset_f1 = df_dataset_f1.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    arch_order = ["bi", "ce"]
    ds_order   = ["pilener", "euroglinerx", "finerweb"]

    archtecture_name = {"bi": "Bi-Encoder", "ce": "Cross-Encoder"}
    eval_benchmarks = {
        "panx": "PAN-X",
        "masakhaner": "MasakhaNER",
        "multinerd": "MultiNERD",
        "multiconer_v1": "MultiCoNER v1",
        "multiconer_v2": "MultiCoNER v2",
        "dynamicner": "DynamicNER",
        "uner": "UNER",
    }
    train_datasets = {
        "pilener": "PileNER",
        "euroglinerx": "EuroGLINER-x",
        "finerweb": "FiNERWeb",
    }

    def rot(s: str) -> str:
        return rf"\rotatebox{{90}}{{{{\centering {s}}}}}"
        
    for m in df_dataset_f1.index.get_level_values("backbone").unique():
        df_filtered = df_dataset_f1.xs(m, level="backbone")
        idx_names = list(df_filtered.index.names)
        arch_lvl, ds_lvl, thr_lvl = idx_names[0], idx_names[1], idx_names[2]
        df_sorted = (
            df_filtered
            .reset_index()
            .assign(
                **{
                    arch_lvl: lambda d: pd.Categorical(d[arch_lvl], categories=arch_order, ordered=True),
                    ds_lvl:   lambda d: pd.Categorical(d[ds_lvl],   categories=ds_order,   ordered=True),
                    thr_lvl:  lambda d: pd.to_numeric(d[thr_lvl].astype(str).str.replace(".json", "", regex=False)),
                }
            )
            .sort_values([arch_lvl, ds_lvl, thr_lvl], ascending=[True, True, True])
            .set_index([arch_lvl, ds_lvl, thr_lvl] + idx_names[3:])
        )
        df_out = df_sorted.rename(columns=eval_benchmarks)
        df_out["Avg."] = df_out.mean(axis=1)
        df_out = (
            df_out
            .rename(index=archtecture_name, level=0)
            .rename(index=train_datasets,    level=1)
        )
        df_out = df_out.rename(index=lambda v: rot(str(v)), level=0)
        df_out = df_out.rename(index=lambda v: rot(str(v)), level=1)
        df_out = df_out.rename(index=lambda v: f"{float(v):.3f}", level=2)

        df_out.index = df_out.index.set_names(["Arc.", "Dataset", r"$\tau$"])
        bench_cols = ["DynamicNER", "MasakhaNER", "MultiCoNER v1", "MultiCoNER v2", "MultiNERD", "PAN-X", "UNER"]
        df_out = df_out[bench_cols + ["Avg."]]
        col_format = "lll|" + "c"*len(bench_cols) + "|r"
        print(
            df_out.to_latex(
                escape=False,
                multirow=True,
                multicolumn=False,
                column_format=col_format,
                float_format="%.3f",
                caption=f"Results for $m={m}$",
            )
        )

def loss_function_results():
    def rot(s: str) -> str:
        return rf"\rotatebox{{90}}{{{{\centering {s}}}}}"

    rows = []
    for result_dir in glob.glob("/vol/tmp/goldejon/ner/paper_results/loss_function_ablation/*"):
        for evaluation_dir in glob.glob(result_dir + "/*"):
            for evaluation_file in glob.glob(evaluation_dir + "/*.json"):
                with open(evaluation_file, "r") as f:
                    data = json.load(f)
                rows.append({
                    "model": result_dir.split("/")[-1],
                    "dataset": evaluation_dir.split("/")[-1],
                    "language": evaluation_file.split("/")[-1].split("_")[0],
                    "threshold": evaluation_file.split("/")[-1].split("_")[1].split(".j")[0],
                    "f1": data["test_metrics"]["micro"]["f1"],
                })

    df = pd.DataFrame(rows)
    df = df[~df['dataset'].str.contains('translated')]
    df['architecture'] = df['model'].apply(lambda x: 'Cross-Encoder' if x.startswith('ce_') else 'Bi-Encoder')
    df['loss'] = df['model'].apply(
        lambda x: (
            'contrastive' if 'contrastive' in x else
            'focal' if 'focal' in x else
            'pos-weight' if 'pos-weight' in x else
            ''
        )
    )
    df['config'] = df['model'].apply(
        lambda x: (
            '$\\alpha=0.5,\\beta=0.5$' if 'contrastive-1' in x else
            '$\\alpha=0.7,\\beta=0.3$' if 'contrastive-2' in x else
            '$\\alpha=0.3,\\beta=0.7$' if 'contrastive-3' in x else
            '$\\alpha=0.25,\\gamma=0.0$' if 'focal-1' in x else
            '$\\alpha=0.50,\\gamma=0.0$' if 'focal-2' in x else
            '$\\alpha=0.75,\\gamma=0.0$' if 'focal-3' in x else
            '$\\alpha=0.5,\\gamma=1.0$' if 'focal-5' in x else
            '$\\alpha=0.75,\\gamma=2.0$' if 'focal-6' in x else
            '$\\lambda=10.0$' if 'pos-weight-1' in x else
            '$\\lambda=100.0$' if 'pos-weight-2' in x else
            ''
        )
    )
    df_dataset_f1 = df.groupby(["architecture", 'loss', 'config', 'dataset', "threshold"]).agg({"f1": "mean"}).reset_index()
    df_dataset_f1 = df_dataset_f1.pivot(index=["architecture", 'loss', 'config', "threshold"], columns="dataset", values="f1")
    df_dataset_f1["Avg."] = df_dataset_f1.mean(axis=1)
    df_dataset_f1 = df_dataset_f1.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    column_names = {
        "architecture": "Arc.",
        "config": "Config",
        "threshold": "$\\tau$",
        "loss": "Loss",
        "panx": "PAN-X",
        "masakhaner": "MasakhaNER",
        "multinerd": "MultiNERD",
        "multiconer_v1": "MultiCoNER v1",
        "multiconer_v2": "MultiCoNER v2",
        "dynamicner": "DynamicNER",
        "uner": "UNER",
    }
    df_dataset_f1 = df_dataset_f1.reset_index()
    df_dataset_f1 = df_dataset_f1.rename(columns=column_names)

    for loss in ["contrastive", "focal", "pos-weight"]:
        subset = df_dataset_f1.copy(deep=True)
        subset = subset[subset["Loss"].str.contains(loss)]

        # keys that should be grouped via LaTeX multirow
        keys = ["Arc.", "Config"]
        if loss != "contrastive":
            keys.append(r"$\tau$")

        # drop non-report columns
        subset = subset.drop(columns=["Loss"])
        subset.columns.name = None  # remove "dataset" corner label if it exists

        # IMPORTANT: multirow only works for index -> move keys to index
        subset = subset.sort_values(keys).set_index(keys)

        # rotate Arc. values (now it is an index level -> rename on that level)
        subset = subset.rename(index=lambda v: rot(str(v)), level="Arc.")

        # column format: key columns (l...) + metrics + Avg
        col_format = ("ll|" if loss == "contrastive" else "lll|") + "c"*7 + "|r"

        print(
            subset.to_latex(
                escape=False,
                index=True,          # must be True for multirow grouping
                multirow=True,
                multicolumn=False,
                column_format=col_format,
                float_format="%.3f",
                caption=f"Detailed results for loss function ablation {loss}",
            )
        )

if __name__ == "__main__":
    main_results()