import os
import numpy as np
import SimpleITK as sitk


def save_for_visaul(exp_dir, epoch, iter, current_visuals):

    real_A = current_visuals['real_A']
    fake_B = current_visuals['fake_B']
    real_B = current_visuals['real_B']

    visual_dir = os.path.join(exp_dir, 'visualise', 'epoch={}'.format(epoch))
    os.makedirs(visual_dir, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(real_A), os.path.join(visual_dir, 'iter={}_realA.nii.gz'.format(iter)))
    sitk.WriteImage(sitk.GetImageFromArray(fake_B), os.path.join(visual_dir, 'iter={}_fakeB.nii.gz'.format(iter)))
    sitk.WriteImage(sitk.GetImageFromArray(real_B), os.path.join(visual_dir, 'iter={}_realB.nii.gz'.format(iter)))


def print_current_errors(epoch, epoch_iter, errors, t):

    print('epoch {} iters {} D_A:{:.4f} G_A:{:.4f} D_B:{:.4f} G_B:{:.4f} Cyc_A:{} Cyc_B:{} time:{}'.format(
        epoch, epoch_iter, errors['D_A'], errors['G_A'], errors['D_B'], errors['Cyc_A'], errors['Cyc_B'], errors['G_B'], t
    ))
    


    
