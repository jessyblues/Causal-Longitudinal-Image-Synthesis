import pandas as pd
import pdb

bio_vars = ['PTAU','ABETA42','TAU'] # 生物标志物
attri_vars = ['Sex', 'Age', 'PTEDUCAT', 'APOE4'] # 人口统计学变量
volume_vars = ['SegVentricles', 'WholeBrain', 'GreyMatter'] # 体积变量

normalize_vars = ['PTAU','ABETA42','TAU','Age','PTEDUCAT','SegVentricles', 'WholeBrain', 'GreyMatter'] # 需要标准化的变量
time_invariant_vars = ['Sex', 'APOE4', 'PTEDUCAT', 'Subject'] # 时间不变变量

csv_path = './dataset/raw_data.csv'  # 数据路径

def get_mean_and_std(): # 计算均值和标准差
    
    def safe_float(x):
        try:
            return float(str(x).replace('<', '').replace('>', ''))
        except:
            return None  # 或 np.nan
    
    def safe_int(x):
        try:
            return int(float(x))
        except:
            return None  # 或 np.nan
    
    df = pd.read_csv(csv_path,
                 usecols=bio_vars+attri_vars+volume_vars+['Subject'],
                 converters={
                     'PTAU': safe_float,
                     'ABETA42': safe_float,
                     'TAU': safe_float,
                     'Age': safe_float,
                     'PTEDUCAT': safe_float,
                     'SegVentricles': safe_float,
                     'WholeBrain': safe_float,
                     'GreyMatter': safe_float,
                     'APOE4': safe_int,
                     'Sex': safe_int,
                     'Subject': str
                 },
                 on_bad_lines='skip'
                )

    df = df.dropna(subset=bio_vars+attri_vars+volume_vars)
    mean_and_std = {}
    for var in normalize_vars:
        try:
            mean_and_std[var] = (df[var].mean(), df[var].std())
        except Exception as e:
            pdb.set_trace()
    
    return mean_and_std

def process_data():  # 读取数据并标准化
    mean_and_std = get_mean_and_std()
    need_values = normalize_vars

    def safe_float(x):
        try:
            return float(str(x).replace('<', '').replace('>', ''))
        except:
            return None

    def safe_int(x):
        try:
            return int(float(x))
        except:
            return None

    # 再次读取时使用相同 converters
    data = pd.read_csv(csv_path,  usecols=bio_vars+attri_vars+volume_vars+['Subject'],
                       converters={
                           'PTAU': safe_float,
                           'ABETA42': safe_float,
                           'TAU': safe_float,
                           'Age': safe_float,
                           'PTEDUCAT': safe_float,
                           'SegVentricles': safe_float,
                           'WholeBrain': safe_float,
                           'GreyMatter': safe_float,
                           'APOE4': safe_int,
                           'Sex': safe_int,
                           'Subject': str
                       },
                       on_bad_lines='skip'
                       )

    # 对标准化变量进行数值转换（确保没有残留字符串）
    for var in need_values:
        data[var] = pd.to_numeric(data[var], errors='coerce')

    # 丢弃 NaN（或可选：填充）
    data = data.dropna(subset=need_values)

    # 标准化
    for var in need_values:
        mean, std = mean_and_std[var]
        data[var] = (data[var] - mean) / std

    return data


def get_longitidinal_data():
    
    new_list = []
    data_df = process_data()
    
    for subject in data_df['Subject'].unique():
        sub_df = data_df[data_df['Subject'] == subject]
        if len(sub_df) <= 1:
           continue
        else:
            sub_df = sub_df.sort_values(by=['Age']) # 按年龄排序
            for i in range(len(sub_df)-1): # 相邻两次随访组成一条数据
                new_row = {k+'_T0':v for k,v in sub_df.iloc[i].items() if k not in time_invariant_vars} # T0时刻的数据
                for k, v in sub_df.iloc[i+1].items():
                    if k not in time_invariant_vars:
                        new_row[k+'_T1'] = v
                
                for k in time_invariant_vars:
                    new_row[k] = sub_df.iloc[i][k] # 时间不变变量只取T0时刻的值
                new_list.append(new_row) # 添加到新数据列表中
    
    new_df = pd.DataFrame(new_list)
    print(f'longitudinal data shape: {new_df.shape}')
    print(f'longitudinal data columns: {new_df.columns}')
    print(f'longitudinal data example: {new_df.iloc[0]}')
    return new_df

if __name__ == '__main__':
    new_df = get_longitidinal_data()
    new_df.to_csv('./dataset/processed_data.csv', index=False)
    mean_and_std = get_mean_and_std()
    df = pd.DataFrame([
    {'var': var, 'mean': mean, 'std': std} 
    for var, (mean, std) in mean_and_std.items()])

    df.to_csv('./dataset/mean_and_std.csv')
