"""Package a frozen single-factor neural component; no training or pricing."""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint, sha256
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN
from src.mentor_dh_pinn.convolution_pinn import ConvolutionPINN


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--component', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    model, info = load_checkpoint(args.component)
    if not isinstance(model, TorchRegularVariancePINN):
        model = TorchRegularVariancePINN.from_mlx(model)
    composite = ConvolutionPINN(model, nodes=96)
    args.out.mkdir(parents=True, exist_ok=False)
    architecture = {k: getattr(model, k) for k in ('factors', 'width', 'depth', 'tau_min',
        'tau_max', 'x_half_width', 'correction_limit', 'residual_blocks')}
    config = {'factors': 2, 'checkpoint_format': 'convolution_price_pinn_float64',
              'component_architecture': architecture, 'quadrature_nodes': 96,
              'component_provenance': info,
              'architecture': 'independent-factor convolution of Single Heston price-PDE PINN',
              'limitations': 'component price PDE, not direct full Double Heston PDE training'}
    (args.out/'config.json').write_text(json.dumps(config, indent=2))
    torch.save(composite.state_dict(), args.out/'model.pt')
    loaded, restored = load_checkpoint(args.out)
    for key, value in composite.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    print(json.dumps({'checkpoint': restored['checkpoint'], 'sha256': restored['sha256']}))


if __name__ == '__main__':
    main()
