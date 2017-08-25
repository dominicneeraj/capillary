import numpy as np
import pandas as pd

df3 = pd.read_csv("training.csv")
df1=pd.read_csv("movies.csv")
# df2=pd.read_csv("data/links.csv")
# df3=pd.read_csv("training.csv")
df4=pd.read_csv("test.csv")

for i in range(10):
    movie_id=int(df3.ix[i][1])
    print(movie_id)
    print((df1.loc[df1['movieId'] == movie_id, 'title']))
print df3.head()

R_df = df3.pivot_table(index = 'userId', columns ='movieId', values = 'rating')

total_num_of_rating=np.count_nonzero(~np.isnan(R_df))
non_rating=R_df.shape[0]*R_df.shape[1]-total_num_of_rating
print(total_num_of_rating)
print(non_rating)

R = R_df.as_matrix()
R_df=R_df.replace(0,0.0001)
p= R_df.replace(np.NaN,0)

class MF():
    
    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            print("Iteration: %d ; error = %.4f" % (i+1, mse))
            c=self.full_matrix()
            np.save('Check_Big'+str(i)+'.npy',c)
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error/971012)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        c=mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)
        np.save('Check_Big_Final.npy',c)
        return c
Rating_mat=np.array(p)

mf = MF(Rating_mat, K=15, alpha=0.001, beta=0.01, iterations=30)
training_process = mf.train()
print()
print("P x Q:")
print(mf.full_matrix())
print()
print("Global bias:")
print(mf.b)
print()
print("User bias:")
print(mf.b_u)
print()
print("Item bias:")
print(mf.b_i)

Rating_Final=mf.full_matrix()
np.max(Rating_Final)

p.reset_index(inplace=True)
list_movies=list(p.columns)
list_movies.remove('userId')
if 'Index' in list_movies:
	list_movies.remove('Index')
if 'index' in list_movies:
	list_movies.remove('index')
list_users=list(p.userId)
df_final_rating=pd.DataFrame(Rating_Final)
df_final_rating.index=list_users
df_final_rating.columns=list_movies
df_final_rating
M=Rating_Final
def movies_corr(users):
    list_item_correlated=dict_item_corr[users]
    return list_item_correlated
def highest_rated(users,no=50):
    df_temp=df_final_rating[df_final_rating.index==users]
    Index_Highest_=df_temp.columns[np.argsort(-df_temp.values, axis=1)[:,:no]]
    Index_Highest_=list(Index_Highest_[0])
    return Index_Highest_
def genre_check(list_movies):
    for Index_ in list_movies:
        temp_var=df1[df1['movieId']==Index_]['genres'].values[0]
        temp_var=temp_var.split('|')
        if (set(temp_var) < set(dict_genres[users])) is False:
            list_movies.remove(Index_)
    return list_movies
import operator
def top_popular_movies(df_nan,no_of_items=20): 
    dict_popular_item={}
    for j in list_movies:
        count=df_nan[j].count()
        mean=np.nanmean(df_nan[j])
        if mean>=4.5:
            dict_popular_item[j]=mean*count
    dict_popular_item=sorted(dict_popular_item.iteritems(), key=operator.itemgetter(1), reverse=True)[:no_of_items]
    list_popular_items=[]
    for key,value in dict_popular_item:
        list_popular_items.append(key)
    return(list_popular_items)   
def already_rated(users):
    DF_temp=R_df[R_df.index==users]
    list_already_rated=[i for i in list_movies if list(DF_temp[i])[0]>=0]
    return list_already_rated

import csv
list_popular_items= top_popular_movies(df_nan,20)
with open('sol_1.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['userId','movieId','rating'])
    for users in df4['userId']:
            print(users)
            Index_Highest_5=[]
            if users==7569560:
                for j in range(5):
#                     Index_num=Index_Highest_[j]
                    Index_Highest_=[7502,599,6777,55071,8012]
                    Rate=4.5
                    writer.writerow([users, Index_Highest_[j], Rate])
                continue
#             list_item_correlated=movies_corr(users)
            list_highest_rated=highest_rated(users,50)
            list_already_rated=already_rated(users)
            list_final=[item for item in list_highest_rated if item not in list_already_rated]
#             for Index_ in Index_Highest_:
#                 temp_var=df1[df1['movieId']==Index_]['genres'].values[0]
#                 temp_var=temp_var.split('|')
#                 if set(temp_var) < set(dict_genres[users]):
#                     Index_Highest_5.append(Index_)
#                 if(len(Index_Highest_5)==5):
#                     break
#             if(len(Index_Highest_5)<5):
#                     for i in range(5):
#                         if len(Index_Highest_5)==5:
#                             break
#                         elif Index_Highest_[i] in Index_Highest_5:
#                             continue
#                         else:
#                             Index_Highest_5.append(Index_Highest_[i])                       
#             print(Index_Highest_5)
            if(len(list_final)<5):
                    for i in range(5):
                        if len(list_final)==5:
                            break
                        elif list_final[i] in list_popular_items:
                            continue
                        else:
                            list_final.append(list_popular_items[i])
            for j in range(5):
                Index_num=list_final[j]
                Rate=list(df_final_rating[df_final_rating.index==users][Index_num])[0]
                if(Rate>5.0):
                	Rate=5
                if(Rate<0.5):
                	Rate=0
                Rate=round(Rate*2)/2
                writer.writerow([users, list_final[j], Rate])

dict_item_corr={}    
j=0
for users in df4['userId']:
    j=j+1
    print(j)
    if users == 7569560:
        value=''
        continue
    DF_temp=R_df[R_df.index==users]
    list_new=[i for i in list_movies if list(DF_temp[i])[0]>=4.5]
    value=[]
    for movies in list_new:
        Index_items_corr=list(cor_matrix[movies][cor_matrix[movies]>0.9].index)
        value=value+Index_items_corr
    dict_item_corr[users]=value
print(dict_item_corr)

cor_matrix=df_nan.corr()
dict_genres={}
j=0
for users in df4['userId']:
    j=j+1
    print(j)
    if users == 7569560:
        value=''
        continue
    DF_temp=R_df[R_df.index==users]
    list_new=[i for i in list_movies if list(DF_temp[i])[0]>=4.0]
    value=[]
    for movies in list_new:
        val=df1[df1['movieId']==movies]['genres'].values[0]
        val=val.split('|')
        for va in val:
            value.append(va)
    dict_genres[users]=value
print(dict_genres)

import csv
with open('sol_2.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['userId','movieId','rating'])
    for users in df4['userId']:
            print(users)
            Index_Highest_5=[]
            if users==7569560:
                for j in range(5):
#                     Index_num=Index_Highest_[j]
                    Index_Highest_=[7502,599,6777,55071,8012]
                    Rate=4.5
                    writer.writerow([users, Index_Highest_[j], Rate])
                continue
            list_item_correlated=movies_corr(users)
            list_highest_rated=highest_rated(users,50)
            list_already_rated=already_rated(users)
            list_popular_items= top_popular_movies(df_nan,20)
            list_final=[item for item in list_highest_rated if item not in list_already_rated]
            list_final_=[]
            for items in list_final:
                if(len(list_item_correlated)>1):
                    if items in list_item_correlated:
                        list_final_.append(items)
                        if(len(list_final_)>=5):
                            break
            list_final=list_final_
#             for Index_ in Index_Highest_:
#                 temp_var=df1[df1['movieId']==Index_]['genres'].values[0]
#                 temp_var=temp_var.split('|')
#                 if set(temp_var) < set(dict_genres[users]):
#                     Index_Highest_5.append(Index_)
#                 if(len(Index_Highest_5)==5):
#                     break
#             if(len(Index_Highest_5)<5):
#                     for i in range(5):
#                         if len(Index_Highest_5)==5:
#                             break
#                         elif Index_Highest_[i] in Index_Highest_5:
#                             continue
#                         else:
#                             Index_Highest_5.append(Index_Highest_[i])                       
#             print(Index_Highest_5)
            if(len(list_final)<5):
                    for i in range(5):
                        if list_popular_items[i] not in list_final:
                            list_final.append(list_popular_items[i])
            for j in range(5):
                Index_num=list_final[j]
                Rate=list(df_final_rating[df_final_rating.index==users][Index_num])[0]
                Rate=round(Rate*2)/2
                writer.writerow([users, list_final[j], Rate])

import csv
with open('sol_3_modified.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['userId','movieId','rating'])
    for users in df4['userId']:
            print(users)
            Index_Highest_5=[]
            if users==7569560:
                for j in range(5):
#                     Index_num=Index_Highest_[j]
                    Index_Highest_=[7502,599,6777,55071,8012]
                    Rate=4.5
                    writer.writerow([users, Index_Highest_[j], Rate])
                continue
            list_item_correlated=movies_corr(users)
#             list_highest_rated=highest_rated(users,50)
            list_already_rated=already_rated(users)
            list_popular_items= top_popular_movies(df_nan,20)
            list_final=[item for item in list_item_correlated if item not in list_already_rated]
            list_after_genre_check=genre_check(list_final)
            list_final=list_after_genre_check
            list_final_=[]
            for items in list_final:
                if(len(list_item_correlated)>1):
                    if items in list_item_correlated:
                        list_final_.append(items)
                        if(len(list_final_)>=5):
                            break
            list_final=list_final_
#             for Index_ in Index_Highest_:
#                 temp_var=df1[df1['movieId']==Index_]['genres'].values[0]
#                 temp_var=temp_var.split('|')
#                 if set(temp_var) < set(dict_genres[users]):
#                     Index_Highest_5.append(Index_)
#                 if(len(Index_Highest_5)==5):
#                     break
#             if(len(Index_Highest_5)<5):
#                     for i in range(5):
#                         if len(Index_Highest_5)==5:
#                             break
#                         elif Index_Highest_[i] in Index_Highest_5:
#                             continue
#                         else:
#                             Index_Highest_5.append(Index_Highest_[i])                       
#             print(Index_Highest_5)
            if(len(list_final)<5):
                    for i in range(5):
                        if list_popular_items[i] not in list_final:
                            list_final.append(list_popular_items[i])
            for j in range(5):
                Index_num=list_final[j]
                Rate=list(df_final_rating[df_final_rating.index==users][Index_num])[0]
                Rate=round(Rate*2)/2
                writer.writerow([users, list_final[j], Rate])
