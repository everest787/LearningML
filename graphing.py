import seaborn as sns
%matplotlib inline

## Distribution
# Histogram
sns.histplot(ad_data['Age'],bins=30);

g = sns.FacetGrid(df,hue="Private",palette='coolwarm')
g = g.map(plt.hist,'Outstate',bins=20)

# JointPlot
sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter') #kind='hex','reg'

# PairPlot
sns.pairplot(tips)
sns.pairplot(tips,hue='sex',palette='coolwarm')

## Categorical
# Barplot
sns.barplot(x='sex',y='total_bill',data=tips) #estimator=np.std

sns.countplot(x='sex',data=tips)

# BoxPlot
sns.boxplot(x="day", y="total_bill", data=tips,palette='rainbow') #hue="smoker"

sns.violinplot(x="day", y="total_bill", data=tips,palette='rainbow') #hue='sex',split=True

## Matrix
# Heatmap
sns.heatmap(tips.corr(),cmap='coolwarm',annot=True)

flights.pivot_table(values='passengers',index='month',columns='year')

#Clustermap
sns.clustermap(pvflights)

## Regressions Plots
# Implot
sns.lmplot(x='total_bill',y='tip',data=tips) #hue='sex'