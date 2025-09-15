from pathlib import Path
from faint.common.models.faint import FAINT
from faint.common.models.vint_ddp import ViNT

def main(ckpt_path, model_type):
    ckpt_path = Path(ckpt_path)
    if model_type == "faint":
        Model = FAINT
    elif model_type == "vint":
        Model = ViNT
    else:
        raise ValueError(f"model_type {model_type} not supported")

    model = Model.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to_torchscript(file_path = ckpt_path.with_suffix(".pt"), method="trace")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--model", type=str, default="faint")
    args = parser.parse_args()
    main(args.ckpt_path, args.model)