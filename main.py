import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def clean_data(self):
        # Remover duplicatas
        self.data.drop_duplicates(inplace=True)
        
        # Preencher valores nulos
        self.data.ffill(inplace=True)

    def filter_data(self, genre=None, min_duration=None, min_year=None, director=None):
        if genre:
            self.data = self.data[self.data['Genre'].str.contains(genre, case=False, na=False)]
        
        if min_duration:
            self.data = self.data[self.data['Runtime'] > min_duration]
        
        if min_year:
            self.data = self.data[self.data['Year'] > min_year]
        
        if director:
            self.data = self.data[self.data['Director'].str.contains(director, case=False, na=False)]

    def normalize_data(self):
        # Agrupar filmes por diretor e gênero
        self.data = self.data.groupby(['Director', 'Genre']).agg({'Runtime': 'sum'}).reset_index()

    def sort_data(self, by='IMDB_Rating', ascending=False):
        if by in self.data.columns:
            self.data = self.data.sort_values(by=by, ascending=ascending)
        else:
            print(f"A coluna {by} não existe no DataFrame.")

    def create_new_columns(self):
        available_columns = self.data.columns
        if 'No_of_Votes' in available_columns and 'IMDB_Rating' in available_columns:
            # Classificar filmes por popularidade e receita
            self.data['Popularity'] = self.data['No_of_Votes'] * self.data['IMDB_Rating']
        else:
            print("As colunas 'No_of_Votes' e/ou 'IMDB_Rating' não existem no DataFrame.")

    def encode_categorical_variables(self):
        available_columns = self.data.columns
        if 'Certificate' in available_columns:
            # One-hot encoding para Certificate
            certificate_ohe = pd.get_dummies(self.data['Certificate'], prefix='Certificate')
            self.data = self.data.drop('Certificate', axis=1).join(certificate_ohe)
            
            # Label encoding para Genre
            genre_le = LabelEncoder()
            self.data['Genre'] = genre_le.fit_transform(self.data['Genre'])
        else:
            print("A coluna 'Certificate' não existe no DataFrame.")

    def discretize_ratings(self):
        # Transformar IMDB_Rating em categorias
        bins = [0, 5, 8, 10]
        labels = ['Low', 'Medium', 'High']
        if 'IMDB_Rating' in self.data.columns:
            self.data['IMDB_Rating_Discretized'] = pd.cut(self.data['IMDB_Rating'], bins=bins, labels=labels)

    def treat_outliers(self):
        # Tratamento de outliers em Gross e IMDB_Rating
        for column in ['Gross', 'IMDB_Rating']:
            if column in self.data.columns:
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR
                self.data[column] = np.where(self.data[column] < lower_limit, lower_limit,
                                             np.where(self.data[column] > upper_limit, upper_limit, self.data[column]))

    def save_results(self, file_name, file_format='csv'):
        if file_format == 'csv':
            self.data.to_csv(file_name, index=False)
        elif file_format == 'txt':
            with open(file_name, 'w') as f:
                f.write(self.data.to_string(index=False))

    def start(self):
        self.clean_data()
        self.filter_data()
        self.normalize_data()
        self.sort_data()
        self.create_new_columns()
        self.encode_categorical_variables()
        self.discretize_ratings()
        self.treat_outliers()

        file_name = 'processed_data.csv'
        self.save_results(file_name)


if __name__ == '__main__':
    file_path = r'C:\Users\fvmichelon\Documents\ProjetoDados_IMDB\imdb_top_1000.csv'
    processor = DataProcessor(file_path)
    processor.start()
