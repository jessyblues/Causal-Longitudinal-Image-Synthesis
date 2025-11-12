import torch
import argparse
import os
import numpy as np
import pickle
import SimpleITK as sitk
from models.GAN_mri_package.model_3D import regression_volume2
from models.GAN_mri_package.utils import get_generator_from_ckpt
from causal_discovery_and_SCM_fitting.ISM import get_volumes_delta_pair

@torch.no_grad()
def inference(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # === 1. Load data ===
    delta_ws, delta_volumes, bl_ws, subject_date_list = \
        get_volumes_delta_pair(csv_pth=args.csv_pth,
                               w_folder=args.w_folder,
                               volumes=[args.volume])

    print(f"Loaded {len(delta_ws)} samples from {args.csv_pth}")

    # === 2. Build model and load checkpoint ===
    dim_w = delta_ws[0].shape[0]
    dim_delta_volume = delta_volumes[0].shape[0]

    model = regression_volume2(
        w_dim=dim_w,
        volume_dim=dim_delta_volume,
        hidden_dims=[int(dim_w/2), int(dim_w/8), int(dim_w/16), int(dim_w/64), int(dim_w/256)]
    ).to(device)

    ckpt = torch.load(args.model_pth, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded model from {args.model_pth}")

    # === 3. Load generator ===
    generator = get_generator_from_ckpt(gan_pth=args.gan_pth, gpu_id=0)
    print(f"Loaded GAN generator from {args.gan_pth}")

    # === 4. Prepare input ===
    delta_input = torch.tensor(np.array(delta_ws), dtype=torch.float, device=device)
    bl_input = torch.tensor(np.array(bl_ws), dtype=torch.float, device=device)

    # === 5. Predict ===
    predict_delta_volume = model(bl_input, delta_input)
    attitude = model.attitude_fcs(bl_input)
    direction = model.direction_model.weight.detach()
    direction = direction / torch.norm(direction)
    predict_delta_w = predict_delta_volume * direction / attitude
    predict_w = predict_delta_w + bl_input

    # === 6. Reconstruct MRI images ===
    print("Generating reconstructed images...")
    for idx, (subject, date) in enumerate(subject_date_list):
        recon_w_idx = predict_w[idx].view(1, 12, -1)
        subject_w_folder = os.path.join(args.w_folder, subject)
        bl_date = sorted(os.listdir(subject_w_folder))[0]
        w_and_noise_file = os.path.join(subject_w_folder, bl_date, 'w_and_noise.pkl')

        if not os.path.exists(w_and_noise_file):
            print(f"Missing noise file for {subject}")
            continue

        with open(w_and_noise_file, 'rb') as f:
            data = pickle.load(f)
        bl_noise = data['noise']
        bl_noise = [torch.tensor(n, device=device).unsqueeze(0).unsqueeze(0)
                    if not torch.is_tensor(n) else n.unsqueeze(0).unsqueeze(0).to(device)
                    for n in bl_noise]

        dst_folder = os.path.join(args.output_folder, subject, date)
        os.makedirs(dst_folder, exist_ok=True)

        image = generator(input=recon_w_idx, noise=bl_noise, step=5, input_is_latent=True).detach()
        image = image.squeeze().cpu()
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))

        image_sitk = sitk.GetImageFromArray(image.numpy())
        sitk.WriteImage(image_sitk, os.path.join(dst_folder, 'reconstructed.nii.gz'))

    print(f"✅ Inference completed. Results saved to: {args.output_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for delta-w regression model')
    parser.add_argument('--csv_pth', required=True, type=str, help='Path to input CSV')
    parser.add_argument('--w_folder', required=True, type=str, help='Folder containing latent w and noise data')
    parser.add_argument('--gan_pth', required=True, type=str, help='Path to pretrained GAN checkpoint')
    parser.add_argument('--model_pth', required=True, type=str, help='Path to trained regression model checkpoint (.model)')
    parser.add_argument('--volume', default='SegVentricles', type=str, help='Target volume type')
    parser.add_argument('--output_folder', default='./inference_results', type=str, help='Folder to save reconstructed MRI images')

    args = parser.parse_args()
    inference(args)
