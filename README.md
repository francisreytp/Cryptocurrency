# Cryptocurrency
### Cryptocurrency Analysis


Our task for this project is to give a report on cryptocurrency. Given some raw information, we classify cryptocurrencies that are trading and has a mining history to create our data set.
The data are standardized, then reduced to three dimensions using PCA.
This dimensions helped us in our valuation of K-means. The K-means is then used to graph and elbow curve which will be used to determine the best K value.


![](/Resources/elbow_curve.png)


From our graph, the elbow is more prominent at K=4. We've set our cluster=4 for our KMeans.
We then formulated a model that created our clustered dataframe.


![](/Resources/fig_c4_scatter.png)


With our filtered cryptocurrency dataframe, we can graph our data.


![](/Resources/fig_c4_3d.png)


We now have a visual representation of how cryptocurrencies can be grouped for further decision making. This report will help our investors in the development of a new investment product.



### Resources
- crypto_data.csv
- Visual Studio
- Jupyter Notebook


