"""Aynı kilitli ağırlıklarda aynı-hedef-zaman-bloğu yorumunu tanısal değerlendir."""

import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import torch
from sklearn.metrics import accuracy_score,roc_auc_score
from kt_project.models import ModelConfig
from kt_project.plkt_temporal import build_temporal_tracer
from kt_project.sequences import InteractionStore,PrefixDataset,collate_prefix,to_device
from kt_project.training import evaluate,prefix_batches,cpu_state,state_digest

RUN=ROOT/"outputs/main/paper_alignment_v2"
OUT=RUN/"interpretations"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    policy=read(ROOT/"configs/inference_interpretation.json")
    prior=read(ROOT/"outputs/method_audit/attention_policy_lock.json")
    assert prior["config_sha256"]==sha(ROOT/"configs/inference_interpretation.json")
    assert prior["implementation_sha256"]==sha(ROOT/"src/kt_project/plkt_temporal.py")
    if (OUT/"results.json").exists():
        stored=read(OUT/"results.json")
        assert stored["complete"] and stored["policy_lock"]==prior
        print(json.dumps({"phase":"attention_interpretation_cached","conditions":len(stored["runs"])}))
        return
    assert read(RUN/"status.json")["status"]=="complete"
    source=read(RUN/"results.json")
    config=read(RUN/"config.json")
    torch.set_num_threads(4)
    device=torch.device(source["device"])
    precision=source["precision"]
    model_config=ModelConfig(**source["model_config"])
    store=InteractionStore(ROOT/"data/processed/interactions.npz")
    validation=PrefixDataset(store,1,model_config.max_history)
    val_batches=prefix_batches(validation,config["evaluation_batch_size"],device,17,config)
    candidates=[row for row in source["runs"] if row["variant"] in policy["methods"]]
    OUT.mkdir(exist_ok=True)
    checkpoints={f"seed_{row['seed']}/{row['variant']}_best.pt":
        sha(RUN/f"seed_{row['seed']}/{row['variant']}_best.pt") for row in candidates}
    save(OUT/"checkpoint_lock.json",{"checkpoints":checkpoints,"policy_lock":prior})
    rows=[]
    def load(row):
        saved=torch.load(RUN/f"seed_{row['seed']}/{row['variant']}_best.pt",map_location="cpu",weights_only=True)
        model=build_temporal_tracer(model_config,row["method"]).to(device)
        model.load_state_dict(saved["model_state"])
        assert state_digest(cpu_state(model))==state_digest(saved["model_state"])
        model.eval().requires_grad_(False)
        return model
    for candidate in candidates:
        folder=OUT/f"seed_{candidate['seed']}"
        folder.mkdir(exist_ok=True)
        model=load(candidate)
        metrics,predictions=evaluate(model,val_batches,device,precision)
        np.savez_compressed(folder/f"{candidate['variant']}_validation_predictions.npz",**predictions)
        rows.append({"seed":candidate["seed"],"method":candidate["method"],"variant":candidate["variant"],
            "fixed_epoch":candidate["best_epoch"],"validation":metrics,"test":None,
            "strict_validation":candidate["validation"],"strict_test":candidate["test"]})
    selected=min((row for row in rows if row["method"]==policy["prototype_method"]),
        key=lambda row:(row["validation"]["bce"],row["seed"]))
    checkpoint=RUN/f"seed_{selected['seed']}/kmpt_best.pt"
    selection={"format":"plkt_method_v2","method":"kmpt","variant":"kmpt","layout":"paper_tail",
        "inference_policy":policy["prototype_policy"],"seed":selected["seed"],"best_epoch":selected["fixed_epoch"],
        "validation_bce":selected["validation"]["bce"],
        "criterion":"minimum block-policy validation BCE among three fixed strict-selected KMPT snapshots",
        "checkpoint":checkpoint.relative_to(ROOT).as_posix(),"checkpoint_sha256":sha(checkpoint),
        "inference_implementation_sha256":sha(ROOT/"src/kt_project/plkt_temporal.py"),
        "test_metrics_used_for_selection":False,"author_code_verified":False,
        "note":"Explicit same-step mask interpretation for working prompt access; strict-token reference retained."}
    # Prototype policy is prescribed in the pre-test policy config; seed uses validation only.
    save(OUT/"prototype_selection.json",selection)
    save(OUT/"selection_before_test.json",{"selection":selection,"new_test_results_available":False})
    test=PrefixDataset(store,2,model_config.max_history)
    test_batches=prefix_batches(test,config["evaluation_batch_size"],device,17,config)
    for row in rows:
        model=load(row)
        metrics,predictions=evaluate(model,test_batches,device,precision)
        folder=OUT/f"seed_{row['seed']}"
        np.savez_compressed(folder/f"{row['variant']}_test_predictions.npz",**predictions)
        with np.load(RUN/f"seed_{row['seed']}/{row['variant']}_test_predictions.npz") as strict:
            np.testing.assert_array_equal(predictions["row_index"],strict["row_index"])
            np.testing.assert_array_equal(predictions["label"],strict["label"])
        assert np.isclose(roc_auc_score(predictions["label"],predictions["probability"]),metrics["auc"],atol=1e-10)
        assert np.isclose(accuracy_score(predictions["label"],predictions["probability"]>=.5),metrics["accuracy"],atol=1e-10)
        row["test"]=metrics
        print(json.dumps({"phase":"temporal_block_test","seed":row["seed"],"method":row["method"],"auc":metrics["auc"]}),flush=True)
    sample=to_device(collate_prefix([validation[int(i)] for i in np.linspace(0,len(validation)-1,16,dtype=int)]),device)
    model=load(selected)
    with torch.inference_mode():
        before=torch.sigmoid(model(sample))
        torch.manual_seed(2026)
        model.prompt_encoder.soft_prompt.add_(torch.randn_like(model.prompt_encoder.soft_prompt)*.5)
        after=torch.sigmoid(model(sample))
    delta=float((before-after).abs().max())
    assert all(sha(RUN/name)==digest for name,digest in checkpoints.items())
    summary={}
    for method in policy["methods"]:
        chosen=[row for row in rows if row["method"]==method]
        summary[method]={metric:{"mean":float(np.mean([row["test"][metric] for row in chosen])),
                                 "sample_sd":float(np.std([row["test"][metric] for row in chosen],ddof=1))}
                         for metric in ("auc","accuracy","bce")}
    result={"complete":True,"policy_lock":prior,"runs":rows,"summary":summary,"prototype_selection":selection,
        "selected_model_prompt_probability_delta":delta,"checkpoint_weights_unchanged":True,
        "same_test_targets_verified":True,"selection_uses_test":False,
        "scope":"diagnostic inference policies on frozen strict-selected snapshots, no per-policy epoch retuning or new training"}
    save(OUT/"results.json",result)
    print(json.dumps({"phase":"attention_interpretation_complete","prototype_seed":selection["seed"],"prompt_delta":delta}),flush=True)


if __name__=="__main__":
    main()
