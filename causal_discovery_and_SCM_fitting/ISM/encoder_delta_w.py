from causal_discovery_and_SCM_fitting.ISM import get_volumes_delta_pair, get_small_deltas, get_volumes_w_pair
import os
import argparse
import random
from models.GAN_mri_package.model_3D import delta_w_encoder, regression_volume, regression_volume2
import numpy as np
import torch.nn.functional as F
import torch
import pdb
from models.GAN_mri_package.utils import get_generator_from_ckpt
import pickle
import SimpleITK as sitk
import matplotlib.pyplot as plt



def train(model, optimizer, training_data, batch_size, device):

    model.train()
    delta_w, delta_volume, bl_ws = training_data

    total_batch = int(np.ceil(len(delta_w)/batch_size))
    loss_record = []

    for batch_idx in range(total_batch):

        if batch_idx == total_batch-1:
            training_delta_w = delta_w[batch_idx*batch_size:]
            training_delta_volume = delta_volume[batch_idx*batch_size:]
            training_bl_ws = bl_ws[batch_idx*batch_size:]
        else:
            training_delta_w = delta_w[batch_idx*batch_size:(batch_idx+1)*batch_size]
            training_delta_volume = delta_volume[batch_idx*batch_size:(batch_idx+1)*batch_size]
            training_bl_ws = bl_ws[batch_idx*batch_size:(batch_idx+1)*batch_size]
        
        #pdb.set_trace()
        
        input = np.hstack((np.array(training_bl_ws), np.array(training_delta_volume)))
        input = torch.tensor(input, device=device, dtype=torch.float)

        target_delta_w = torch.tensor(np.array(training_delta_w), device=device, dtype=torch.float)
        #target_norm_delta_w = target_delta_w/torch.norm(target_delta_w, dim=1, keepdim=True).repeat(1, 12*128)

        #cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')

        predict_delta_w = model(input)
        
        #loss = cos_loss(predict_delta_w, target_delta_w, torch.ones((predict_delta_w.shape[0]), device=device))
        loss = F.l1_loss(target_delta_w, predict_delta_w)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_record.append(loss.item())
    
    return np.mean(loss_record)


def validate(model, val_data, batch_size, device):

    model.eval()
    delta_w, delta_volume, bl_ws = val_data

    total_batch = int(np.ceil(len(delta_w)/batch_size))
    loss_record = []

    for batch_idx in range(total_batch):

        if batch_idx == total_batch-1:
            val_delta_w = delta_w[batch_idx*batch_size:]
            val_delta_volume = delta_volume[batch_idx*batch_size:]
            val_bl_ws = bl_ws[batch_idx*batch_size:]
            
        else:
            val_delta_w = delta_w[batch_idx*batch_size:(batch_idx+1)*batch_size]
            val_delta_volume = delta_volume[batch_idx*batch_size:(batch_idx+1)*batch_size]
            val_bl_ws = bl_ws[batch_idx*batch_size:(batch_idx+1)*batch_size]

        if len(val_bl_ws) != len(val_delta_volume):
            pdb.set_trace()
        
        input = np.hstack((np.array(val_bl_ws), np.array(val_delta_volume)))
        input = torch.tensor(input, device=device, dtype=torch.float)

        target_delta_w = torch.tensor(np.array(val_delta_w), device=device, dtype=torch.float)

        with torch.no_grad():
            predict_delta_w = model(input)
        
        #loss = F.l1_loss(predict_delta_w, target_delta_w)
        #cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')

        predict_delta_w = model(input)
        
        #loss = cos_loss(predict_delta_w, target_delta_w, torch.ones((predict_delta_w.shape[0]), device=device))
        loss = F.l1_loss(target_delta_w, predict_delta_w)

        #pdb.set_trace()
        loss_record.append(loss.item())
    
    return np.mean(loss_record)

def test_recon_img(test_data, device, save_folder,
                   gan_pth='/home1/yujiali/cf_mri_2/StyleGAN/exp_1/checkpoint/172000.model', 
                   w_folder='/home1/yujiali/cf_mri_2/Encoder_GAN/search_exp/exp1',
                   model_pth=None, model=None):

    delta_w, delta_volume, bl_ws, subject_date_list = test_data
    generator = get_generator_from_ckpt(gan_pth=gan_pth, gpu_id=0)

    dim_w = delta_w[0].shape[0]
    dim_delta_volume = delta_volume[0].shape[0]
    
    input = np.hstack((np.array(bl_ws), np.array(delta_volume)))
    input = torch.tensor(input, device=device, dtype=torch.float)

    if model is None:
        model = delta_w_encoder(in_dim=dim_w+dim_delta_volume, num_layers=12).to(device)
        ckpt = torch.load(model_pth, map_location=device)
        model.load_state_dict(ckpt['generator'])

    with torch.no_grad():
        predict_delta_w = model(input)
    
    predict_w = (predict_delta_w + torch.tensor(np.array(bl_ws), dtype=torch.float, device=device))
    #predict_w = predict_delta_w

    for idx, (subject, date) in enumerate(subject_date_list):

        recon_w_idx = predict_w[idx].view(1, 12, -1)

        subject_w_folder = os.path.join(w_folder, subject)
        bl_date = sorted(os.listdir(subject_w_folder))[0]
        
        w_and_noise_file = os.path.join(subject_w_folder, bl_date, 'w_and_noise.pkl')

        if os.path.exists(w_and_noise_file):
            f_read = open(w_and_noise_file, 'rb+')
            try:
                dict2 = pickle.load(f_read, encoding = 'utf-8')
            except:
                print('something wrong when open {}'.format(w_and_noise_file))
                continue
        
        bl_noise = dict2['noise']
        
        if not torch.is_tensor(bl_noise[0]):
            bl_noise = [torch.tensor(noise_layer, device=device).unsqueeze(0).unsqueeze(0) for noise_layer in bl_noise]
        else:
            bl_noise = [noise_layer.unsqueeze(0).unsqueeze(0).to(device) for noise_layer in bl_noise]
        
        dst_folder = os.path.join(save_folder, subject, date)
        os.makedirs(dst_folder, exist_ok=True)
        
        #pdb.set_trace()
        with torch.no_grad():
            image = generator(input=recon_w_idx, noise=bl_noise, step=5, input_is_latent=True).detach()
            
        image = image.squeeze().cpu()
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        #image = (torch.clip(image, min=0))/torch.max(image)
        #image = (image_torch)/torch.max(image)


        th = 35/256
        mask = image > th 
            
        masked_img = image * mask
        image_sitk = sitk.GetImageFromArray(image.numpy())

        sitk.WriteImage(image_sitk, os.path.join(dst_folder, 'rec.nii.gz'))
    

def train_encoder_delta_w(args, delta_ws, delta_volumes, bl_ws, subject_date_list):



    total_sample_number = len(delta_ws)
    training_number = int(total_sample_number*(1-args.test_ratio))
    
    training_data = delta_ws[:training_number], delta_volumes[:training_number], bl_ws[:training_number]
    val_data = delta_ws[training_number:], delta_volumes[training_number:], bl_ws[training_number:]

    print('total:{}, training:{}, test:{}'.format(total_sample_number, training_number, total_sample_number-training_number))

    dim_w = delta_ws[0].shape[0]
    dim_delta_volume = delta_volumes[0].shape[0]
    
    device = torch.device('cuda:0')
    encoder_model = delta_w_encoder(in_dim=dim_w+dim_delta_volume, num_layers=6).to(device)

    #pdb.set_trace()
    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    min_loss = 1 
    best_model = None

    for epoch in range(args.max_epoch):

        training_loss = train(encoder_model, optimizer, training_data, batch_size=args.batch_size, device=device)
        #print('epoch {}, training loss {}'.format(epoch, training_loss))

        if (epoch+1) % args.test_every == 0:
            val_loss = validate(encoder_model, val_data=val_data, batch_size=args.batch_size, device=device)
            print('epoch {}, train loss {} val loss {}'.format(epoch, training_loss, val_loss))
            
            
            if val_loss<min_loss:
                model_dict = {'model':encoder_model.state_dict()}
                min_loss = val_loss
                best_model = encoder_model

    
    torch.save(model_dict, os.path.join(args.exp_folder, 'checkpoint', 
                                        'best_model_l1_loss={}.model'.format(round(min_loss, 4))))
    return best_model
    


def regession_volume_on_w(args, data, gan_pth):

    delta_ws, delta_volumes, bl_ws, subject_date_list = data

    total_sample_number = len(delta_ws)
    training_number = int(total_sample_number*(1-args.test_ratio))
    
    training_data = delta_ws[:training_number], delta_volumes[:training_number], bl_ws[:training_number], subject_date_list[:training_number]
    val_data = delta_ws[training_number:], delta_volumes[training_number:], bl_ws[training_number:], subject_date_list[training_number:]
    test_data = delta_ws[-100:-80], delta_volumes[-100:-80], bl_ws[-100:-80], subject_date_list[-100:-80]

    print('total:{}, training:{}, test:{}'.format(total_sample_number, training_number, total_sample_number-training_number))

    dim_w = (delta_ws[0].shape[0])
    dim_delta_volume = delta_volumes[0].shape[0]
    device = torch.device('cuda:0')
    #pdb.set_trace()
    regression_model = regression_volume2(w_dim=dim_w, volume_dim=dim_delta_volume, 
                                         hidden_dims=[int(dim_w/2), int(dim_w/8), int(dim_w/16), int(dim_w/64), int(dim_w/256)]).to(device)
    #regression_model = regression_volume(w_dim=dim_w, volume_dim=dim_delta_volume, 
    #                                     hidden_dims=[]).to(device)
    

    optimizer = torch.optim.Adam(regression_model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    min_loss = 500
    best_model = None

    for epoch in range(args.max_epoch):

        #input = np.hstack((np.array(val_bl_ws), np.array(val_delta_volume)))
        delta_input = torch.tensor(np.array(training_data[0]), 
                                        device=device, dtype=torch.float)
        bl_input = torch.tensor(np.array(training_data[2]), 
                                        device=device, dtype=torch.float)

        target_delta_volume = torch.tensor(np.array(training_data[1]), device=device, dtype=torch.float)

        predict_delta_volume = regression_model(bl_input, delta_input)
        #predict_delta_volume = regression_model(delta_input)
        training_loss = F.l1_loss(target_delta_volume, predict_delta_volume)
        
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        if (epoch+1) % args.test_every == 0:
            regression_model.eval()
            
            with torch.no_grad():
                delta_input = torch.tensor(np.array(val_data[0]),
                                      device=device, dtype=torch.float)
                bl_input = torch.tensor(np.array(val_data[2]),
                                      device=device, dtype=torch.float)
            target_delta_volume = torch.tensor(np.array(val_data[1]), device=device, dtype=torch.float)

            predict_delta_volume = regression_model(bl_input, delta_input)
            #predict_delta_volume = regression_model(delta_input)
            eval_loss = F.l1_loss(target_delta_volume, predict_delta_volume)
            error_percentage = predict_delta_volume - target_delta_volume

            #delta_volumes, bl_ws
            error_percentage = torch.mean(error_percentage, dim=0).squeeze().tolist()

            
            print('epoch:{}, training loss:{}, eval loss:{}'.format(epoch, training_loss.item(), eval_loss.item()))
                  #torch.mean(torch.abs(target_delta_volume), dim=0).squeeze().tolist())
            if eval_loss.item() < min_loss:
                min_loss = eval_loss.item()
                best_model = regression_model
    
    with torch.no_grad():
        delta_input = torch.tensor(np.array(val_data[0]),
                                      device=device, dtype=torch.float)
        bl_input = torch.tensor(np.array(val_data[2]),
                                      device=device, dtype=torch.float)
        target_delta_volume = torch.tensor(np.array(val_data[1]), device=device, dtype=torch.float)
        predict_delta_volume = best_model(bl_input, delta_input)
        #predict_delta_volume = best_model(delta_input)
    
    target_delta_volume = target_delta_volume.cpu().numpy().squeeze()
    predict_delta_volume = predict_delta_volume.cpu().numpy().squeeze()

    #pdb.set_trace()
    arrIndex = target_delta_volume.argsort()
    target_delta_volume = target_delta_volume[arrIndex]
    predict_delta_volume = predict_delta_volume[arrIndex]
    subject_data_list_val = np.array(val_data[3])
    #pdb.set_trace()
    subject_data_list_val = subject_data_list_val[arrIndex]
    #pdb.set_trace()
    plt.figure()
    
    plt.plot(target_delta_volume)
    plt.plot(predict_delta_volume)
    plt.legend(['real', 'predict'])
    plt.title('ventrcle')
    plt.savefig(os.path.join(args.exp_folder,'regression_model2.jpg'))

    
    
    delta_w, delta_volume, bl_ws, subject_date_list = test_data
    generator = get_generator_from_ckpt(gan_pth=gan_pth, gpu_id=0)
    
    delta_input = torch.tensor(np.array(delta_w),
                                      device=device, dtype=torch.float)
    bl_input = torch.tensor(np.array(bl_ws),
                                      device=device, dtype=torch.float)
    target_delta_volume = torch.tensor(np.array(delta_volume), 
                                       device=device, dtype=torch.float)

    dim_w = delta_w[0].shape[0]
    dim_delta_volume = delta_volume[0].shape[0]
    
    attitude = best_model.attitude_fcs(bl_input)
    direction = best_model.direction_model.weight.detach()
    direction = direction/torch.norm(direction)
    predict_delta_w = target_delta_volume * direction / attitude

    predict_w = predict_delta_w + bl_input
    
    
    for idx, (subject, date) in enumerate(subject_date_list):

        recon_w_idx = predict_w[idx].view(1, 12, -1)

        subject_w_folder = os.path.join(args.w_folder, subject)
        bl_date = sorted(os.listdir(subject_w_folder))[0]
        
        w_and_noise_file = os.path.join(subject_w_folder, bl_date, 'w_and_noise.pkl')

        if os.path.exists(w_and_noise_file):
            f_read = open(w_and_noise_file, 'rb+')
            try:
                dict2 = pickle.load(f_read, encoding = 'utf-8')
            except:
                print('something wrong when open {}'.format(w_and_noise_file))
                continue
        
        bl_noise = dict2['noise']
        
        if not torch.is_tensor(bl_noise[0]):
            bl_noise = [torch.tensor(noise_layer, device=device).unsqueeze(0).unsqueeze(0) for noise_layer in bl_noise]
        else:
            bl_noise = [noise_layer.unsqueeze(0).unsqueeze(0).to(device) for noise_layer in bl_noise]
        
        save_folder = os.path.join(args.exp_folder, 'test', 'regression_model2')
        dst_folder = os.path.join(save_folder, subject, date)
        os.makedirs(dst_folder, exist_ok=True)
        
        with torch.no_grad():
            image = generator(input=recon_w_idx, noise=bl_noise, step=5, input_is_latent=True).detach()
            
        image = image.squeeze().cpu()
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))


        image_sitk = sitk.GetImageFromArray(image.numpy())

        sitk.WriteImage(image_sitk, os.path.join(dst_folder, 'rec.nii.gz'))
    
    return best_model
    
    
def get_direction(training_data, val_data, args, device):

    delta_w1, delta_volume1, bl_ws1, subject_date_list1 = training_data
    delta_w2, delta_volume2, bl_ws2, subject_date_list2 = val_data

    delta_w1 = torch.tensor(np.array(delta_w1), device=device, dtype=torch.float)
    delta_w2 = torch.tensor(np.array(delta_w2), device=device, dtype=torch.float)
    dim_w = delta_w1[0].shape[0]

    norm_delta_w1 = delta_w1/torch.norm(delta_w1, dim=1, keepdim=True).repeat(1, 12*128)
    norm_delta_w2 = delta_w2/torch.norm(delta_w2, dim=1, keepdim=True).repeat(1, 12*128)

    projection = torch.nn.Parameter(torch.rand(dim_w, dim_w).to(device))
    

    cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
    optimizer = torch.optim.Adam(params=[projection], lr=1e-3)

    for idx in range(norm_delta_w1.shape[0]-1):

        print(torch.dot(norm_delta_w1[idx], norm_delta_w1[idx+1]))
        print()
        #pdb.set_trace()

    for epoch in range(args.max_epoch):
        
        #pdb.set_trace()
        projected_delta_w1 = torch.mm(norm_delta_w1, projection)
        norm_projected_delta_w1 = projected_delta_w1/torch.norm(projected_delta_w1, dim=1, keepdim=True).repeat(1, 12*128)
        mean_direction = torch.mean(norm_projected_delta_w1, dim=0)
        loss = cos_loss(mean_direction.repeat(delta_w1.shape[0], 1), norm_projected_delta_w1, torch.ones((delta_w1.shape[0]), device=device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % args.test_every:

            projected_delta_w2 = torch.mm(norm_delta_w2, projection)
            norm_projected_delta_w2 = projected_delta_w2/torch.norm(projected_delta_w2, dim=1, keepdim=True).repeat(1, 12*128)
            mean_direction = torch.mean(norm_projected_delta_w2, dim=0)
            eval_loss = cos_loss(mean_direction.repeat(delta_w2.shape[0], 1), norm_projected_delta_w2, torch.ones((delta_w2.shape[0]), device=device))


            print('epoch:{}, training consine loss:{}, test loss:{}, det:{}'.format(epoch, loss.item(), eval_loss.item(), torch.det(projection)))

def check_delta_w_angle(data):
    
    device = torch.device('cuda:0')
    delta_ws, delta_volumes, bl_ws, subject_date_list = data
    previous_subject = ''
    delta_ws_same_person = []
    cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
    
    for idx, (subject, date) in enumerate(subject_date_list):
        
        if subject != previous_subject:
            
            if len(delta_ws_same_person) > 1:
                delta_w1 = torch.tensor(np.array(delta_ws_same_person), device=device, dtype=torch.float)
                norm_delta_w1 = delta_w1/torch.norm(delta_w1, dim=1, keepdim=True).repeat(1, 12*128)

                mean_direction = torch.mean(norm_delta_w1, dim=0)
                loss = cos_loss(mean_direction.repeat(delta_w1.shape[0], 1), norm_delta_w1, torch.ones((delta_w1.shape[0]), device=device))
                print(loss.item())
                pdb.set_trace()

            previous_subject = subject
            delta_ws_same_person = [delta_ws[idx]]
        else:
            delta_ws_same_person += [delta_ws[idx]]

def check_if_w_contains_info(args, data, device):

    ws, volumes, subject_date_list = data

    total_sample_number = len(ws)
    training_number = int(total_sample_number*(1-args.test_ratio))
    
    training_data = ws[:training_number], volumes[:training_number], subject_date_list[:training_number]
    val_data = ws[training_number:], volumes[training_number:], subject_date_list[training_number:]


    print('total:{}, training:{}, test:{}'.format(total_sample_number, training_number, total_sample_number-training_number))

    dim_w = (ws[0].shape[0])
    dim_delta_volume = volumes[0].shape[0]
    regression_model = regression_volume(w_dim=dim_w, volume_dim=dim_delta_volume, 
                                         hidden_dims=[int(dim_w/2), int(dim_w/8), int(dim_w/16), int(dim_w/64), int(dim_w/256)]).to(device)

    

    optimizer = torch.optim.Adam(regression_model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    min_loss = 500
    best_model = None

    for epoch in range(args.max_epoch):

        #input = np.hstack((np.array(val_bl_ws), np.array(val_delta_volume)))
        #delta_input = torch.tensor(np.array(training_data[0]), 
        #                                device=device, dtype=torch.float)
        input = torch.tensor(np.array(training_data[0]), 
                                        device=device, dtype=torch.float)

        #pdb.set_trace()
        target_volume = torch.tensor(np.array(training_data[1]), device=device, dtype=torch.float)

        predict_volume = regression_model(input)
        training_loss = F.l1_loss(predict_volume, target_volume)
        
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        if (epoch+1) % args.test_every == 0:
            regression_model.eval()
            
            with torch.no_grad():
                input = torch.tensor(np.array(val_data[0]),
                                      device=device, dtype=torch.float)
            target_volume = torch.tensor(np.array(val_data[1]), device=device, dtype=torch.float)

            predict_volume = regression_model(input)

            eval_loss = F.l1_loss(target_volume, predict_volume)

            
            print('epoch:{}, training loss:{}, eval loss:{}'.format(epoch, training_loss.item(), eval_loss.item()))
                  #torch.mean(torch.abs(target_delta_volume), dim=0).squeeze().tolist())
            if eval_loss.item() < min_loss:
                min_loss = eval_loss.item()
                best_model = regression_model
    
    with torch.no_grad():
        input = torch.tensor(np.array(val_data[0]),
                                      device=device, dtype=torch.float)

        target_volume = torch.tensor(np.array(val_data[1]), device=device, dtype=torch.float)
        predict_volume = best_model(input)
    
    target_volume = target_volume.cpu().numpy().squeeze()
    predict_volume = predict_volume.cpu().numpy().squeeze()

    #pdb.set_trace()
    arrIndex = target_volume.argsort()
    target_volume = target_volume[arrIndex]
    predict_volume = predict_volume[arrIndex]
    subject_data_list_val = np.array(val_data[2])

    subject_data_list_val = subject_data_list_val[arrIndex]
    #pdb.set_trace()
    plt.figure()
    
    plt.plot(target_volume)
    plt.plot(predict_volume)
    plt.legend(['real', 'predict'])
    plt.title('Hippo')
    plt.savefig(os.path.join(args.exp_folder,'Hippo_regression_deleta_model.jpg'))

    
    return best_model






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training delta w encoder')
    

    parser.add_argument('--csv_pth', default='./dataset/causal_longitudinal_data_only_volume.csv', type=str) 
    parser.add_argument('--w_folder', default='./ImageEncoder/exp', type=str)
    parser.add_argument('--test_ratio', default=0.3, type=float)
    parser.add_argument('--exp_folder', default='./causal_discovery_and_SCM_fitting/ISM/exp', type=str)
    parser.add_argument('--gan_pth', default=None, type=str)
    parser.add_argument('--volume', default='SegVentricles', type=str)

    parser.add_argument('--max_epoch', default=1000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--test_every', default=10, type=int)

    args = parser.parse_args()



    delta_ws, delta_volumes, bl_ws, subject_date_list = \
          get_volumes_delta_pair(csv_pth=args.csv_pth, 
                                 w_folder=args.w_folder,
                                 volumes=[args.volume])


    total_sample_number = len(delta_ws)
    training_number = int(total_sample_number*(1-args.test_ratio))

    training_data = delta_ws[:training_number], delta_volumes[:training_number], bl_ws[:training_number], subject_date_list[:training_number]
    val_data = delta_ws[training_number:], delta_volumes[training_number:], bl_ws[training_number:], subject_date_list[training_number:]

    device = torch.device('cuda:1')
    best_model = regession_volume_on_w(args, data=(delta_ws, delta_volumes, bl_ws, subject_date_list), 
                                      gan_pth=args.gan_pth)
    
    torch.save({'model':best_model.state_dict()}, f'{args.exp_folder}/best.model')




    

    

