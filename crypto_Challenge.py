#%%
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hvplot.pandas
import plotly.express as px 

#%%
####################### Data Preprocessing ######################
#%%
# Load data
crypto_df = pd.read_csv("Resources/crypto_data.csv")
crypto_df.head(10)
#%%
# Number of data
len(crypto_df)

#%%
# Column type
crypto_df.dtypes

#%%
# Removing crypto that are not trading
trading_crypto_df = crypto_df[crypto_df['IsTrading'] == True]
trading_crypto_df.head(10)

#%%
# Number of currency that are trading
len(trading_crypto_df)

# %%
# Counting crypto that don’t have algorithm defined
trading_crypto_df['Algorithm'].isnull().sum()

#%%
# No crypto has a null value
trading_wAlgorithm_df = trading_crypto_df
trading_wAlgorithm_df.head(5)

#%%
# Remove the IsTrading column
trading_wAlgorithm_df = trading_wAlgorithm_df.drop(columns = ['IsTrading'])
trading_wAlgorithm_df.head(5)

#%%
# Find null values
for column in trading_wAlgorithm_df:
    print(f"Column {column} has {trading_wAlgorithm_df[column].isnull().sum()} null values")

#%%
# Removing cryptoc with at least one null value
notnull_crypto_df = trading_wAlgorithm_df.dropna()
print(f"{len(notnull_crypto_df)} cryptocurrency with no null values")

#%%
# Crypto with no coines mined
notnull_crypto_df[notnull_crypto_df['TotalCoinsMined']== 0]

#%%
# Removing crypto with no coins mined
ableToMine_crypto_df = notnull_crypto_df[notnull_crypto_df['TotalCoinsMined'] != 0]
len(ableToMine_crypto_df)
print(f"There are {len(ableToMine_crypto_df)} instances of crypto mining")


#%%
# Store names of crypto on a DataFrame, using "Unnamed: 0" as index
coins_name = pd.DataFrame(ableToMine_crypto_df[['Unnamed: 0','CoinName']])
coins_name.set_index('Unnamed: 0', drop = True, inplace = True)
# coins_name.index.names = ['']
coins_name.head()

#%%
# Making sure we didnt lose any data...
print(f"We have {len(coins_name)} number of data")

#%%
# Removing CoinName column
clean_crypto_df = ableToMine_crypto_df.drop(columns = ['CoinName'])
clean_crypto_df.head()

#%%
# Inspecting data types
clean_crypto_df.dtypes

#%%
# Changing data type for TotalCoinSupply
clean_crypto_df['TotalCoinSupply'] = clean_crypto_df['TotalCoinSupply'].astype('float')

#%%
# Double check if changes applied
# clean_crypto_df.dtypes

# %%
# Create dummies variables for text features, and store results to DataFrame
X = pd.get_dummies(clean_crypto_df[['Algorithm','ProofType']])
X.head()

#%%
# Standardize the data from X
scale_model = StandardScaler()
scaled_X = scale_model.fit_transform(X)
scaled_X

#%%
####################   PCA   #######################

#%%
# Reducing X DataFrame Dimensions Using PCA to 3 features
pca = PCA(n_components=3)
X_pca = pca.fit_transform(scaled_X)
print(f'pca ratio - {pca.explained_variance_ratio_}')

#%%
# Explained variance
pca.explained_variance_

#%%
pcs_df = pd.DataFrame(X_pca, index=clean_crypto_df["Unnamed: 0"], columns=['PC 1','PC 2','PC 3'])
# pcs_df.index.names = ['']
pcs_df.head(10)

#%%
################### Clustering Using K-means  ###############

#%%
# Graph elbow curve to find best value for K,
#   X-axis is K, y-axis is inertia
inertia_list = list()
k_value = list(range(1,11))

for k in k_value:
    k_model = KMeans(n_clusters=k, random_state=1)
    k_model.fit(pcs_df)
    inertia_list.append(k_model.inertia_)

# DataFrame for plotting
elbow_df = pd.DataFrame({'K': k_value, 'Inertia': inertia_list})

#%%
# Graph the Elbow curve
elbow_curve = elbow_df.hvplot.line(x = 'K', y = 'Inertia', xticks = k_value, title='Elbow Curve')
elbow_curve

#%%
# From our graph, the elbow is more prominent at K=4. We will set our cluster=4 for our KMeans

#%%
# KMeans algorithm
model = KMeans(n_clusters=4, random_state=1)
predictions = model.fit_predict(pcs_df)


#%%
# Combining clean_crypto_df, pcs_df, and coins_name
clustered_df = clean_crypto_df.merge(pcs_df, on = 'Unnamed: 0')
clustered_df = clustered_df.merge(coins_name, on = 'Unnamed: 0')

# Integrate algorithm in the DataFrame
clustered_df['Class'] = model.labels_

clustered_df.set_index('Unnamed: 0', drop = True, inplace = True)
clustered_df.index.names = ['']
clustered_df.head(10)


#%%
#################### Visualizing Results #####################

#%%
# Scatter plot 3D
fig_c4_3d = px.scatter_3d(clustered_df, x= 'PC 1', y='PC 2',z='PC 3',
                    color='Class', symbol='Class', hover_name='CoinName',
                    hover_data=['Algorithm'])
fig_c4_3d.update_layout(legend = {'x':0,'y':1})
fig_c4_3d

# %%
# hvplot Table
crypto_table = clustered_df.hvplot.table(columns = ['CoinName', 'Algorithm', 
                                    'ProofType', 'TotalCoinSupply', 
                                    'TotalCoinsMined', 'Class'], width =500)

crypto_table


#%%
# Scatter plot 2D
fig_c4_scatter = clustered_df.hvplot.scatter(x="TotalCoinsMined", y="TotalCoinSupply",
                                by = 'Class', hover_cols = ['CoinName'])

fig_c4_scatter



#%%
