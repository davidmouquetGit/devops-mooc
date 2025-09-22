import gdown
import pandas as pd
import os


def download_fromdrive(file_id:str,data_path:str) -> str:

    url = f"https://drive.google.com/uc?id={file_id}"

    # Téléchargement du fichier
    raw_data_path = os.path.join(data_path,'raw')
    os.makedirs(raw_data_path,exist_ok=True)

    raw_file = os.path.join(raw_data_path,'raw_data.csv')

    gdown.download(url, raw_file, quiet=False)
    return raw_file

def clean_data(raw_data_file:str, data_path:str) -> None:

    from sklearn.model_selection import train_test_split


    df = pd.read_csv(raw_data_file,sep=';',decimal=",")
    col_to_keep = ['PU_MOY_F2','METAL','LARG_S_E5','D_EPF_1','D_EPF_2']

    df = df[col_to_keep]
    df["METAL"] = df["METAL"].astype(str)
    df.dropna(inplace=True)

    data_train, data_test = train_test_split(df, test_size=0.2,random_state=42)

    data_path = os.path.join(data_path,'cleaned')
    os.makedirs(data_path,exist_ok=True)

    data_train.iloc[0:1000].to_csv(os.path.join(data_path,"train_data.csv"),sep=";",decimal=".")
    data_test.iloc[0:1000].to_csv(os.path.join(data_path,"test_data.csv"),sep=";",decimal=".")

def main():

    file_id = "1vvWPwW4_uFE5BPFODbHxHMXeBQ4cAmmS"
    raw_data_file = download_fromdrive(file_id=file_id,data_path='data')
    clean_data(raw_data_file=raw_data_file,data_path='data')

if __name__ == '__main__':
    main()

