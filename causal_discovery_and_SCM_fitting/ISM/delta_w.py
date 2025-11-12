import os
import pickle
import csv
import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

def get_subject_date_imageid(img_folder):

    subject_date_imageid = {}
    subject_date = {}

    subjects = os.listdir(img_folder)
    
    for subject in sorted(subjects):
        subject_folder = os.path.join(img_folder, subject)
        if not os.path.isdir(subject_folder):
            continue
        dates = os.listdir(subject_folder)
        
        for date in sorted(dates):

            date_folder = os.path.join(subject_folder, date)
            if not os.path.isdir(date_folder):
                continue
            
            mni_file = os.listdir(date_folder)[0]
            Image_ID = mni_file.split('_brain')[0]
            #pdb.set_trace()

            subject_date_imageid[(subject, date)] = Image_ID
            
            if subject in subject_date.keys():
                subject_date[subject].append(date)
            else:
                subject_date[subject] = [date]
    
    return subject_date_imageid, subject_date


def get_w_from_file(w_folder):

    subject_date_w_pair = {}
    subjects = os.listdir(w_folder)
    
    for subject in subjects:
        subject_folder = os.path.join(w_folder, subject)
        
        if not os.path.isdir(subject_folder):
            continue
        
        dates = os.listdir(subject_folder)
        for date in sorted(dates):
            date_folder = os.path.join(subject_folder, date)
            w_and_noise_file = os.path.join(date_folder, 'w_and_noise.pkl')

            if os.path.exists(w_and_noise_file) and len(os.listdir(date_folder))==2:
                f_read = open(w_and_noise_file, 'rb+')
                try:
                    dict2 = pickle.load(f_read, encoding = 'utf-8')
                except:
                    print('something wrong when open {}'.format(w_and_noise_file))
                    continue
                w = dict2['w'].reshape(-1, )
                subject_date_w_pair[(subject, date)] = w
        
    return subject_date_w_pair

def get_id_volume(csv_pth, volumes=None):

    id_volumes_pair = {}
    if type(volumes) == str:
        volumes=[volumes]
    
    if volumes is None:
        volumes = ['Ventricles','Hippocampus','WholeBrain','Entorhinal','Fusiform','MidTemp','Grey Matter','White Matter']

    with open(csv_pth, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)
        for row in reader:
            volume_vector = [row[v] for v in volumes]

            volume_vector_ = [float(v) for v in volume_vector if v!= '']
            if '' in volume_vector:
                continue
            id = row['Image Data ID']
            id_volumes_pair[id] = volume_vector_

    return id_volumes_pair



def get_volumes_delta_pair(csv_pth, w_folder, img_folder, volumes=None, mean_and_std=None):

    id_volumes_pair = get_id_volume(csv_pth, volumes=volumes) ##complete

    subject_date_imageid, subject_date = get_subject_date_imageid(img_folder)
    subject_date_w_pair = get_w_from_file(w_folder) 

    delta_ws = []
    delta_volumes = []
    bl_ws = []
    subject_date_list = []

    volume_mean, volume_std = mean_and_std
    

    for subject in subject_date.keys():
        
        dates = subject_date[subject]
        skip_subject = False

        for idx, date in enumerate(dates):
            
            if skip_subject:
                continue
            
            if idx == 0:
                bl_id = subject_date_imageid[(subject, date)]
                
                if (subject, date) not in subject_date_w_pair.keys():
                    skip_subject = True
                    continue
                if bl_id not in id_volumes_pair.keys():
                    skip_subject = True
                    continue
                
                bl_w = subject_date_w_pair[(subject, date)]
                #pdb.set_trace()
                bl_volume = (np.array(id_volumes_pair[bl_id]) - volume_mean) / volume_std
               # bl_volume = np.array(id_volumes_pair[bl_id])

            else:
                id = subject_date_imageid[(subject, date)]

                if (subject, date) not in subject_date_w_pair.keys():
                    continue
                if id not in id_volumes_pair.keys():
                    continue

                delta_w = np.array(subject_date_w_pair[(subject, date)]) - np.array(bl_w)
                new_volume = (np.array(id_volumes_pair[id]) - volume_mean) / volume_std
                #new_volume = np.array(id_volumes_pair[id])
                
                delta_volume = (new_volume - bl_volume)
                #pdb.set_trace() 

                delta_ws += [delta_w]
                delta_volumes += [delta_volume]
                bl_ws += [bl_w]
                subject_date_list += [(subject, date)]
    
    
    return delta_ws, delta_volumes, bl_ws, subject_date_list

def get_volumes_w_pair(csv_pth, w_folder, volumes=None, normalise=True, mean_and_std=None):
    
    id_volumes_pair = get_id_volume(csv_pth, volumes=volumes) ##complete
    #pdb.set_trace()
    if type(w_folder) == list:
        subject_date_w_pair = {}
        for w_folder_ in w_folder:
            subject_date_w_pair.update(get_w_from_file(w_folder_))
    else:
        subject_date_w_pair = get_w_from_file(w_folder) ##not complete
    #pdb.set_trace()

    
    subject_date_imageid, subject_date = get_subject_date_imageid() ##complete
    #pdb.set_trace()

    volumes_list = []
    ws = []
    subject_date_list = []

    if normalise:
        volume_mean, volume_std = mean_and_std

    for subject in subject_date.keys():
        
        dates = subject_date[subject]
        #skip_subject = False

        for idx, date in enumerate(dates):
            
            id = subject_date_imageid[(subject, date)]

            if (subject, date) not in subject_date_w_pair.keys():
                continue
            if id not in id_volumes_pair.keys():
                continue

            w = np.array(subject_date_w_pair[(subject, date)])
            
            new_volume = (np.array(id_volumes_pair[id]) - volume_mean) / volume_std if normalise else np.array(id_volumes_pair[id])
            
            ws += [w]
            volumes_list += [new_volume]
            subject_date_list += [(subject, date)]

    
    return ws, volumes_list, subject_date_list


def get_small_deltas(delta_ws=None, delta_volumes=None, bl_ws=None, subject_date_list=None, csv_pth=None, 
                     w_folder='/home1/yujiali/cf_mri_2/Encoder_GAN/search_exp/exp1', 
                     volumes=None):

    if csv_pth is not None:
        delta_ws, delta_volumes, bl_ws, subject_date_list = get_volumes_delta_pair(csv_pth, w_folder, volumes=volumes)
    
    small_delta_ws = []
    small_delta_volumes = []
    small_bl_ws = []
    previous_subject = ''
    
    for idx, delta_w in enumerate(delta_ws):
        
        subject = subject_date_list[idx][0]

        small_delta_w = delta_ws[idx] - delta_ws[idx-1] if subject == previous_subject else delta_ws[idx]
        small_delta_volume = delta_volumes[idx] - delta_volumes[idx-1] if subject == previous_subject else delta_volumes[idx]
        small_bl_w = bl_ws[idx] + delta_ws[idx-1] if subject == previous_subject else bl_ws[idx]
        
        small_delta_volumes.append(small_delta_volume)
        small_delta_ws.append(small_delta_w)
        small_bl_ws.append(small_bl_w)

        previous_subject = subject 
    
    return small_delta_ws, small_delta_volumes, small_bl_ws, subject_date_list

def get_individual_change(csv_pth, volumes, ratio=False, group=None):

    subject_volumes_pair = {}
    subject_date_pair = {}
    
    if volumes is None:
        volumes = ['Ventricles','Hippocampus','WholeBrain','Entorhinal','Fusiform','MidTemp','Grey Matter','White Matter']

    volumes_list = []
    subejct_list = []
    date_list = []


    with open(csv_pth, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)
        for row in reader:
            try:
                volume_vector = [float(row[v]) / float(row['WholeBrain']) for v in volumes] if ratio else [float(row[v]) for v in volumes]
            except Exception as e:
                continue
            #pdb.set_trace()
            subject = row['Subject']
            date = row['Acq Date']
            group_ = row['Group']
            if '/' in date:
                month, day, year = tuple(date.split('/'))
            else:
                year, month, day = tuple(date.split('-'))
            
            if group is None:
                subejct_list.append(subject)
                date_list.append('{}-{:0>2d}-{}'.format(year, int(month), day))
                volumes_list.append(volume_vector)
            else:
                if group_ in group:
                    subejct_list.append(subject)
                    date_list.append('{}-{:0>2d}-{}'.format(year, int(month), day))
                    volumes_list.append(volume_vector)
    
    subejct_list = np.array(subejct_list)
    date_list = np.array(date_list)
    volumes_list = np.array(volumes_list)
    #pdb.set_trace()
    arrIndex = subejct_list.argsort()
    subejct_list = subejct_list[arrIndex]
    date_list = date_list[arrIndex]
    volumes_list = volumes_list[arrIndex]

    previous_subject = ''

    for idx, subject in enumerate(subejct_list):
        
        if subject != previous_subject:

            if idx > 0 :
                subject_date_list = np.array(subject_date_list)
                subject_volumes_list = np.array(subject_volumes_list)
                #if subject_date_list.shape[0] == 1:
                #    continue
                arrIndex = subject_date_list.argsort()
                subject_date_list = subject_date_list[arrIndex]
                subject_volumes_list = subject_volumes_list[arrIndex]

                subject_volumes_pair[previous_subject] = subject_volumes_list.squeeze()
                subject_date_pair[previous_subject] = subject_date_list

                #pdb.set_trace()



            subject_date_list = [date_list[idx]]
            subject_volumes_list = [[volumes_list[idx]]]
            previous_subject = subject
        
        else:
            
            subject_date_list += [date_list[idx]]
            subject_volumes_list += [[volumes_list[idx]]]

        
    return subject_date_pair, subject_volumes_pair

def draw_change(csv_pth, num=10, save_folder='/home1/yujiali/cf_mri_2/causal', 
                volume='Ventricles', ratio=False, group=None):

    subject_date_pair, subject_volumes_pair = get_individual_change(csv_pth, [volume], ratio, group=None)
    
    
    list_ = list(subject_date_pair.keys())
    random.shuffle(list_)


    plt.figure()
    plt.title(volume)
    #pdb.set_trace()
    for i in range(num):

        plt.plot(subject_volumes_pair[list_[i]].astype('float'))
    
    plt.legend(list_[:num])
    plt.savefig(os.path.join(save_folder, '{}_trend.jpg'.format(volume)))



def get_subject_date_volumes(csv_pth='/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/seg_results.csv', volume_='Grey Matter', ratio=True):

    #if volumes is None:
    #    volumes = ['Ventricles','Hippocampus','WholeBrain','Entorhinal','Fusiform','MidTemp','Grey Matter','White Matter']
    subject_date_volume = {}

    with open(csv_pth, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)
        for row in reader:
            volume = float(row[volume_]) / float(row['WholeBrain']) if ratio else float(row[volume_]) 
            subject = row['Subject']
            date = row['Acq Date']
            subject_date_volume[(subject, date)] = volume
        
        #volume_mean = np.mean(np.array(volume_value), axis=0)
        #volume_std = np.std(np.array(volume_value), axis=0)

    #return volume_mean, volume_std
    return subject_date_volume

        




if __name__ == '__main__':
    
    idx = 0
    subject_date_volume = get_subject_date_volumes(csv_pth='/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/info_merged3_all.csv', volume_='White Matter')
    for k, v in subject_date_volume.items():
        if v < 0.25:
            print(k, v)
            idx += 1  
    print(idx)





    












            