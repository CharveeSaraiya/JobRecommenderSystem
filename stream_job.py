import streamlit as st
import pandas as pd
from rake_nltk import Rake
import operator
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import wordcloud as w
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, make_scorer, classification_report, accuracy_score
import scikitplot as skplt
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
import warnings

# Load the data
@st.cache_resource
def load_data():
    job_postings = 'DATASET\DATASET\dice_com-job_us_sample.csv'
    df_job = pd.read_csv(job_postings)
    return df_job

df_job = load_data()

# Streamlit UI
st.title("Job Recommendation and Analysis")

# Display the first 5 lines of the loaded data
st.subheader("Preview of Data")
st.write(df_job.head())

# Data Cleaning
st.subheader("Data Cleaning")

# Remove unnecessary columns
columns_to_remove = ['postdate', 'shift', 'site_name', 'uniq_id', 'jobid']
df_job = df_job.drop(columns=columns_to_remove)

# Remove rows with missing values
df_job = df_job.dropna()

# Further data cleaning

# Display the cleaned data
st.write(df_job.head())



#check to see if we have any missing values at all.
st.subheader("Check  Missing Values")
st.write(df_job.isnull ().values.any ())
# since the above results is true, check number of missing values in each row
st.subheader("Missing Values in each row")
st.write(df_job.isna().sum())


# In[ ]:


# Total number of missing values
st.subheader("Total Number of Missing Values")
st.write(df_job.isnull().sum().sum())


# In[ ]:


#drop all rows that have any NaN values
df_job.dropna(inplace=True)

##check that there is no more na
st.subheader("Job Wise Sum of NA")
st.write(df_job.isna().sum())


# In[ ]:


# Get index of all rows in skills that contains the value "Null"
jobsNull = df_job[df_job["skills"]=="Null"]. index
# Get index of all rows in skills that contains the value "Please see job description"
jobsDesc1 = df_job[df_job["skills"]=="Please see job description"].index
# Get index of all rows in skills that contains the value (See Job Description)
jobsDesc2 = df_job[df_job["skills"]==" (See Job Description)"]. index
# Get index of all rows in skills that contains the value "SEE BELOW*
jobsDesc3 = df_job[df_job["skills"]=="SEE BELOW"]. index
# Get index of all rows in skills that contains the value "Telecommuting not available Travel not required"
jobsDesc4 = df_job[df_job["skills"]=="Telecommuting not available Travel not required"]. index
# Get index of all rows in skills that contains the value "Refer to Job Description"
jobsDesc5 = df_job[df_job["skills"]=="Refer to Job Description"].index
# Get index of all rows in skills that contains the value "Please see Required Skills"
jobsDesc6 = df_job[df_job["skills"]=="Please see Required Skills"].index
# drop rows of index
df_job.drop(jobsNull, inplace=True)
df_job.drop(jobsDesc1, inplace=True)
df_job.drop(jobsDesc2, inplace=True)
df_job.drop(jobsDesc3, inplace=True)
df_job.drop(jobsDesc4, inplace=True)
df_job.drop(jobsDesc5, inplace=True)
df_job.drop(jobsDesc6, inplace=True)


# In[ ]:


#Print info of Dataframe
#st.write(df_job.info())
st.subheader("Information of DataSet")
st.write(df_job.info())

df_job["jobtitle"].value_counts ()[:5]


# In[ ]:


job = []
stopwordsList = []
cleanJobs = []
# Get the stopwords and store in list
with open("DATASET\DATASET\stopwords.txt", 'r', encoding="utf-8") as f:
    for word in f:
        word = word.split('\n')
        stopwordsList.append(word[0])

# Tokenizing and Removing stop words from jobtitle

from nltk.tokenize import word_tokenize
import nltk

#nltk.download('punkt')
# Convert all words to lower case and change the shortform
for i in df_job['jobtitle'].values:
  jobs = i. lower ()
  jobs = jobs.replace("QA", "Quality Assurance")
  jobs = jobs.replace("sr", "Senior")
  jobs = jobs.replace("jr", "Junior")
  jobs = jobs.replace("qm", "Quality Manager")
  job.append (jobs)

# tokenize and remove the words from stop words
for j in job:
    text_tokens = word_tokenize(j)
    tokens_without_sw = [f for f in text_tokens if not f in stopwordsList]
    cleanJobs .append(' '.join(tokens_without_sw))


# In[ ]:


# Add the newly cleaned job title into the dataframe
df_job['clean_jobtitle'] = cleanJobs
st.subheader("Job wise count")
st.text(df_job.head(5))

qty = df_job["clean_jobtitle"].value_counts()[:5].tolist ()
label = df_job["clean_jobtitle"].value_counts () [:5]. index. tolist ()
st.text(qty)
st.text("Top 5 Popular Jobs :" + str(label))


# In[ ]:


def addlabels(x,y):
  for i in range(len(x)):
    plt.text(i, y[i], y[i], ha = 'center')

## initializing the labels
skillslabel = label
jobQty = qty

st.title("Top 5 High Demand Jobs")

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(skillslabel, jobQty, color=['purple', 'red', 'green', 'blue', 'orange'])

# Add labels
for i in range(len(skillslabel)):
    ax.text(i, jobQty[i], jobQty[i], ha='center')

# Set the title and labels
ax.set_title("Top 5 High Demand Jobs")
ax.set_xlabel("Name of Jobs")
ax.set_ylabel("Quantity")

# Display the chart in the Streamlit app
st.pyplot(fig)


# **2. Most Used Skills**

# In[ ]:


skillsTokenized = []
stopwordsskills = []




# Get the stopwords and store in list
with open("DATASET\DATASET\stopwords.txt", 'r', encoding="utf-8") as f:
    for word in f:
      word.lower()
      word = word.split('\n')

      stopwordsskills.append(word[0])

for k in df_job['skills'].values:
  k = str(k).split(', ')
  #remove stopwords from skills
  skillstokens_without_sw = [f for f in k if not f.lower() in stopwordsskills]
  for j in skillstokens_without_sw:
    skillsTokenized.append(j)

#put the cleaned skills into a new dataframe
df = pd.DataFrame({'skills':skillsTokenized})


# In[ ]:


#Get the top 5 skills
qtySkills = df["skills"].value_counts().tolist()
labelSkills = df["skills"].value_counts().index.tolist()
st.write("Top 5 skills mostly needed \n" + str(df["skills"].value_counts()[:5]))


# In[ ]:


import wordcloud as w
import numpy as np
import matplotlib.pyplot as plt

lskills = labelSkills
frequencies = qtySkills

#Wordcloud asks for a string, and we have tried seperating the terms with ',' and '-'

# d = dict(zip(lskills,frequencies))
# wordcloud = w.WordCloud(collocations = False, random_state=1,background_color='White', width=3000,height = 2000).generate_from_frequencies(d)

# plt.imshow(wordcloud,interpolation = 'bilinear')
# plt.axis("off")
# plt.figure(figsize = (3000,3000))
# st.pyplot.show()



st.title("Word Cloud Visualization")

# Create a Word Cloud
d = dict(zip(lskills, frequencies))
wordcloud = w.WordCloud(
    collocations=False,
    random_state=1,
    background_color='white',
    width=800,
    height=400
).generate_from_frequencies(d)

# Display the Word Cloud
st.image(wordcloud.to_array())





# Data Analysis
st.header("Data Analysis")

# Top 5 most demanded jobs
st.subheader("Top 5 Most Demanded Jobs")
top_jobs = df_job['clean_jobtitle'].value_counts()[:5]
st.write(top_jobs)




vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform(df_job['clean_jobtitle'].values)
print(X.shape)
analyze = vectorizer.build_analyzer()
# print( 'Job titles', analyze(str(jobtitle)))
# print ('Document transform', X. toarray ())
# print(vectorizer.get_feature_names())
features = vectorizer.get_feature_names_out()
# indices = zip(*X.nonzero)
# for row, column in indices:
#   print(' (%d, %s) %f' %(row, features [column], X[row, column]))


# # **4. Clustering using K - Means**

# 4.1 getting Optimize cluster using elbow method

# In[ ]:


from sklearn.cluster import KMeans
# Using the elbow method to find the optimal number of clusters
# Within Cluster Sum of Squares(WCSS)
wcss =[]
for i in range (1,15):
    kmeans = KMeans(n_clusters = i,init='k-means++',random_state = 42,max_iter = 600,n_init = 1)
    kmeans.fit(X)
    #inertia method returns wcss for the model
    wcss.append(kmeans.inertia_)


st.title('Elbow Method for K-Means Clustering')

# Plotting the graph
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 15), y=wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

# Display the plot within the Streamlit app
st.pyplot(plt)


true_k = 7  # Set the desired number of clusters
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=600, n_init=1, random_state=42)
pred = model.fit_predict(X)
order_centeroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names_out()


# In[ ]:


from sklearn.cluster import KMeans
from sklearn. decomposition import PCA

sklearn_pca = PCA(n_components = 2)

Y_sklearn = sklearn_pca.fit_transform(X. toarray())
kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=600, n_init=1, random_state=42)
fitted = kmeans.fit (Y_sklearn)
prediction = kmeans.predict (Y_sklearn)

st.title('K-Means Clustering')

# Plot the scatter points
plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=prediction, s=50, cmap='viridis')
ceriters = kmeans.cluster_centers_
# Plot the cluster centers
plt.scatter(ceriters[:, 0], ceriters[:, 1], c='black', s=300, alpha=0.6)

# Add labels to the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Cluster Centers')

# Display the plot within the Streamlit app
st.pyplot(plt)



from sklearn.metrics import silhouette_score
st.text ('KMeans Scaled Silhouette Score: {}'.format(silhouette_score(X, model.labels_, metric = 'euclidean' )))


# # In[ ]:


def get_top_keywords (data, clusters, labels, n_terms):
  df = pd.DataFrame(data.todense()).groupby(clusters).mean()
  for i,r in df.iterrows():
    st.text('\ncluster {}'.format(i))
    st.text(','.join([labels[t] for t in np.argsort(r)[-n_terms: ]]))

get_top_keywords(X, pred, features, 10)


label = []
for i in df_job['clean_jobtitle'].values:
      vec = vectorizer.transform([i])
      pred = model.predict (vec)
      if pred == 0:
          label.append("Project Management")
      elif pred == 1:
          label.append("Frontend")
      elif pred == 2:
          label.append("Devops/Software Engineer")
      elif pred == 3:
          label.append("Business Solution Consultant")
      elif pred == 4:
          label.append("Cloud Architect/Network")
      elif pred == 5:
          label.append("Analyst")
      else:
          label.append("IT Business Management")

df_job['Label'] = label

st.write(df_job.head(5))



jobSkills = []
for i in df_job['skills']:
    jobSkills.append(i.lower ())

Xclass = vectorizer.fit_transform(jobSkills)
#Split data into test and train. Test size 20% Train Size 80%
X_train,X_test,y_train,y_test = train_test_split(Xclass, label, test_size=0.2, random_state=42)


# # In[ ]:
# st.title('Logistic Regression Model')

# # Obtain the best C range
# Cparamrange = [0.1,0.5,0.8,1,2]
# trainAcc = []
# testAcc = []
# for i in Cparamrange:
#     lrg = LogisticRegression(penalty = 'l2', C = i,random_state = 42)
#     lrg.fit(X_train,y_train)
#     lrg_predtrain=lrg.predict(X_train)
#     lrg_predtest=lrg.predict(X_test)
#     trainacc = accuracy_score (y_train, lrg_predtrain)
#     testacc = accuracy_score(y_test,lrg_predtest)
#     trainAcc.append(trainacc)
#     testAcc. append(testacc)

# # plt.plot(Cparamrange, trainAcc, 'ro-', Cparamrange, testAcc, 'bv-.')
# # plt.legend(['Training Accuracy', 'Test Accuracy'])
# # plt.xlabel('Number of C')
# # plt.ylabel('Accuracy')

# st.line_chart({'Training Accuracy': trainAcc, 'Test Accuracy': testAcc})
# plt.title('Logistic Regression Model Accuracy')
# plt.xlabel('Number of C')
# plt.ylabel('Accuracy')
# # Display the plot within the Streamlit app
# st.pyplot(plt)


# # using the best c param range
# lrg = LogisticRegression(penalty = 'l2', C = 0.1, random_state = 42)
# lrg.fit(X_train,y_train)
# lrg_pred=lrg.predict(X_test)
# lrg_acc = accuracy_score(y_test,lrg_pred)
# st.subheader ("Accuracy of Logistic Regression: " + str(lrg_acc))
# st.text(classification_report (y_test, lrg_pred))


# # In[ ]:



# #plot confusion matrix
# # skplt.metrics.plot_confusion_matrix(
# #       y_test,
# #       lrg_pred,
# #       x_tick_rotation=90,
# #       figsize=(6,5))


# fig, ax = plt.subplots(figsize=(6, 5))
# skplt.metrics.plot_confusion_matrix(y_test, lrg_pred, x_tick_rotation=90, ax=ax)

# # Display the confusion matrix in the Streamlit app
# st.pyplot(fig)

# from sklearn.neighbors import KNeighborsClassifier
# #get_ipython().run_line_magic('matplotlib', 'inline')

# numNeighbors = [1, 5, 6,7, 8, 10, 15, 20, 25, 30,35,40,50]
# trainAcc = []
# testAcc = []

# for k in numNeighbors:
#     clf1 = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
#     clf1.fit(X_train, y_train)
#     Y_predTrain = clf1.predict(X_train)
#     Y_predTest = clf1.predict(X_test)
#     trainAcc.append(accuracy_score(y_train, Y_predTrain))
#     testAcc.append(accuracy_score(y_test, Y_predTest))

# plt.plot(numNeighbors, trainAcc, 'ro-', numNeighbors, testAcc,'bv--')
# plt.legend([ 'Training Accuracy', 'Test Accuracy'])
# plt.xlabel('Number of neighbors')
# plt.ylabel( 'Accuracy')
# st.pyplot(plt)


# knn = KNeighborsClassifier(n_neighbors=50, metric='minkowski', p=2)
# knn. fit (X_train, y_train)
# knn_pred = clf1.predict(X_test)
# knn_acc = accuracy_score (y_test, knn_pred)
# st.subheader("Accuracy of KNN: " + str (knn_acc))
# st.text(classification_report(y_test, knn_pred))


# # In[ ]:


# # skplt.metrics.plot_confusion_matrix(
# # y_test,
# # knn_pred,
# # x_tick_rotation=90,
# # figsize= (6,5))
# fig, ax = plt.subplots(figsize=(6, 5))
# skplt.metrics.plot_confusion_matrix(y_test, knn_pred, x_tick_rotation=90, ax=ax)

# # Display the confusion matrix in the Streamlit app
# st.pyplot(fig)

# # # **5.3 Decision tree**

# # In[ ]:


# maxdepths = [2,3,4,5,6,7,8,9,10, 15, 20, 25,30,35,40,45,50] # 17 different depths
# trainAccuracy = np.zeros(len(maxdepths))
# testAccuracy = np.zeros(len(maxdepths))
# index = 0
# for depth in maxdepths:
#     clf2 = DecisionTreeClassifier (max_depth=depth)
#     clf2 = clf2.fit (X_train, y_train)
#     Y_predTrain = clf2.predict (X_train)
#     Y_predTest = clf2.predict(X_test)
#     trainAccuracy[index] = accuracy_score(y_train, Y_predTrain)
#     testAccuracy[index] = accuracy_score(y_test, Y_predTest)
#     index += 1

# # Plot training and test accuracies

# plt.plot (maxdepths, trainAccuracy, 'ro-' , maxdepths, testAccuracy , 'bv--')
# plt.legend(['Training Accuracy', 'Test Accuracy'])
# plt.xlabel('Max depth')
# plt.ylabel('Accuracy')
# st.pyplot(plt)




# dt = DecisionTreeClassifier(max_depth = 10)
# dt2 = dt.fit(X_train, y_train)
# dt_pred = dt2.predict(X_test)
# dt_acc = accuracy_score(y_test,dt_pred)
# st.subheader("Accuracy of Decision Trees: "+ str(dt_acc))
# st.text(classification_report(y_test,dt_pred))


# # In[ ]:


# # skplt.metrics.plot_confusion_matrix(
# #     y_test,
# #     dt_pred,
# #     x_tick_rotation = 90,
# #     figsize =(6,5)
# # )

# fig, ax = plt.subplots(figsize=(6, 5))
# skplt.metrics.plot_confusion_matrix(y_test, dt_pred, x_tick_rotation=90, ax=ax)

# # Display the confusion matrix in the Streamlit app
# st.pyplot(fig)

# In[ ]:


from sklearn import svm


# In[ ]:


Csvm = [0.1,0.5,0.8,1,1.5, 2,2.5,3,3.5]
trainAcc = []
testAcc = []
for c in Csvm:
    modelsvm = svm.SVC(C= c, gamma= 1, kernel = 'rbf')
    svmfit = modelsvm.fit(X_train,y_train)
    Y_predTrain = modelsvm.predict(X_train)
    Y_predTest = modelsvm.predict (X_test)
    trainAcc.append(accuracy_score(y_train, Y_predTrain))
    testAcc.append(accuracy_score(y_test, Y_predTest))

plt.plot(Csvm, trainAcc,'ro-', Csvm, testAcc, 'bv--')
plt. legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of C')
plt.ylabel('Accuracy')
st.pyplot(plt)




svm = svm.SVC(C= 5, gamma= 1, kernel= 'rbf' ,probability=True)
svmfit = svm.fit(X_train,y_train)
svm_predictions = svmfit.predict (X_test)
svm_acc = accuracy_score(y_test,svm_predictions)
st.subheader("Accuracy of SVM:" + str(svm_acc))
st.text(classification_report (y_test, svm_predictions))


# In[ ]:


# skplt.metrics.plot_confusion_matrix(
#   y_test,
#   svm_predictions,
#   x_tick_rotation=90,
#   figsize= (6,5))

# fig, ax = plt.subplots(figsize=(6, 5))
# skplt.metrics.plot_confusion_matrix(y_test, svm_predictions, x_tick_rotation=90, ax=ax)

# # Display the confusion matrix in the Streamlit app
# st.pyplot(fig)

# st.title('Model Accuracy')

# # Define the labels and model accuracy. Round the model accuracy to 2 decimal places.
# labels = ("Logistic Regression", "KNeighbors", "Decision Tree", "Support Vector Machines")
# modelsAccuracy = [round(lrg_acc, 2), round(knn_acc, 2), round(dt_acc, 2), round(svm_acc, 2)]

# # Create a bar chart
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.bar(labels, modelsAccuracy, color=['purple', 'red', 'green', 'blue'])

# # Add value labels
# for i, acc in enumerate(modelsAccuracy):
#     ax.text(i, acc, str(acc), ha='center', va='bottom', fontsize=12)

# # Set plot title and labels
# plt.title("Model Accuracy")
# plt.xlabel("Name of Model")
# plt.ylabel("Accuracy")

# # Display the plot within the Streamlit app
# st.pyplot(fig)





# st.title('ROC Curves')

# # Predict probabilities
# lrg_prob = lrg.predict_proba(X_test)[::, 1]
# knn_prob = knn.predict_proba(X_test)[::, 1]
# dt_prob = dt.predict_proba(X_test)[::, 1]
# svm_prob = svm.predict_proba(X_test)[::, 1]

# # ROC curve for models
# fpr_lrg, tpr_lrg, thresh_lrg = roc_curve(y_test, lrg_prob, pos_label='Business Solution Consultant')
# fpr_knn, tpr_knn, thresh_knn = roc_curve(y_test, knn_prob, pos_label='Business Solution Consultant')
# fpr_dt, tpr_dt, thresh_dt = roc_curve(y_test, dt_prob, pos_label='Business Solution Consultant')
# fpr_svm, tpr_svm, thresh_svm = roc_curve(y_test, svm_prob, pos_label='Business Solution Consultant')

# # ROC curve for tpr = fpr
# random_probs = [0 for i in range(len(y_test))]
# P_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label='Business Solution Consultant')

# # Plot ROC curves
# fig, ax = plt.subplots()
# ax.plot(fpr_lrg, tpr_lrg, linestyle='--', color='orange', label='Logistic Regression')
# ax.plot(fpr_knn, tpr_knn, linestyle='--', color='green', label='KNN')
# ax.plot(fpr_dt, tpr_dt, linestyle='--', color='red', label='Decision Tree')
# ax.plot(fpr_svm, tpr_svm, linestyle='--', color='purple', label='SVM')
# ax.plot(P_fpr, p_tpr, linestyle='--', color='blue')

# # Set title, labels, and legend
# plt.title('ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='best')

# # Display the plot within the Streamlit app
# st.pyplot(fig)


# st.title('ROC Curves')

# # Predict probabilities
# lrg_prob = lrg.predict_proba(X_test)[::, 1]
# knn_prob = knn.predict_proba(X_test)[::, 1]
# dt_prob = dt.predict_proba(X_test)[::, 1]
# svm_prob = svm.predict_proba(X_test)[::, 1]

# # ROC curve for models
# fpr_lrg, tpr_lrg, thresh_lrg = roc_curve(y_test, lrg_prob, pos_label='Business Solution Consultant')
# fpr_knn, tpr_knn, thresh_knn = roc_curve(y_test, knn_prob, pos_label='Business Solution Consultant')
# fpr_dt, tpr_dt, thresh_dt = roc_curve(y_test, dt_prob, pos_label='Business Solution Consultant')
# fpr_svm, tpr_svm, thresh_svm = roc_curve(y_test, svm_prob, pos_label='Business Solution Consultant')

# # ROC curve for tpr = fpr
# random_probs = [0 for i in range(len(y_test))]
# P_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label='Business Solution Consultant')

# # Plot ROC curves
# fig, ax = plt.subplots()
# ax.plot(fpr_lrg, tpr_lrg, linestyle='--', color='orange', label='Logistic Regression')
# ax.plot(fpr_knn, tpr_knn, linestyle='--', color='green', label='KNN')
# ax.plot(fpr_dt, tpr_dt, linestyle='--', color='red', label='Decision Tree')
# ax.plot(fpr_svm, tpr_svm, linestyle='--', color='purple', label='SVM')
# ax.plot(P_fpr, p_tpr, linestyle='--', color='blue')

# # Set title, labels, and legend
# plt.title('ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='best')

# # Display the plot within the Streamlit app
# st.pyplot(fig)



# labelData = df_job[df_job['Label'] == "Frontend"]
# skillsClass = []
# for index, row in labelData.iterrows():
#     skills = [row['skills']]
#     skillstokens_without_sw = [f for f in skills if not f.lower () in stopwordsskills]
#     for j in skillstokens_without_sw:
#         skillsClass.append(j)

# # put the cleaned skills into a new dataframe
# df_frontend = pd.DataFrame({'skills':skillsClass})
# qtySkills = df_frontend["skills"].value_counts().tolist()
# labelSkills = df_frontend["skills"].value_counts().index.tolist ()

# st.title('Word Cloud')

# # Combine labels and frequencies into a dictionary
# lskills = labelSkills
# frequencies = qtySkills
# d = dict(zip(lskills, frequencies))

# # Generate a word cloud from the dictionary
# wordcloud = w.WordCloud(collocations=False, background_color='white', width=3000, height=2000).generate_from_frequencies(d)

# # Display the word cloud in the Streamlit app
# st.image(wordcloud.to_array(), use_container_width=True)

# # Optionally, you can add some text or explanations below the word cloud
# st.markdown("Above is a word cloud representing the most used skills.")



# labelData = df_job[df_job['Label'] == "Analyst"]
# skillsClass = []
# for index, row in labelData.iterrows() :
#     skills = [row['skills']]
#     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsskills]
#     for j in skillstokens_without_sw:
#         skillsClass.append(j)
# # put the cleaned skills into a new dataframe
# df_analyst = pd.DataFrame({'skills':skillsClass})

# qtySkills = df_analyst["skills"].value_counts().tolist ()
# labelSkills = df_analyst["skills"].value_counts().index.tolist ()

# # Create a Streamlit app
# st.title('Word Cloud')

# # Combine labels and frequencies into a dictionary
# lskills = labelSkills
# frequencies = qtySkills
# d = dict(zip(lskills, frequencies))

# # Generate a word cloud from the dictionary
# wordcloud = w.WordCloud(collocations=False, background_color='white', width=3000, height=2000).generate_from_frequencies(d)

# # Display the word cloud in the Streamlit app
# st.image(wordcloud.to_array(), use_container_width=True)

# # Optionally, you can add some text or explanations below the word cloud
# st.markdown("Above is a word cloud representing the most used skills.")



# **7.1.3 Busines Solution Consultation**

# In[ ]:


# labelData = df_job[df_job['Label'] == "Business Solution Consulatant"]
# skillsClass = []
# for index,row in labelData.iterrows() :
#     skills = [row[ 'skills']]
#     skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsskills]
#     for j in skillstokens_without_sw:
#         skillsClass.append(j)
# # put the cleaned skills into a new dataframe
# df_consultant = pd.DataFrame ({'skills':skillsClass})
# qtySkills = df_consultant["skills"].value_counts().tolist()
# labelSkills = df_consultant["skills"].value_counts().index.tolist()

# lskills = labelSkills
# frequencies = qtySkills

# # Combine the labels and frequencies into a dictionary
# d = dict(zip(lskills, frequencies))

# # Create a Streamlit app
# st.title("Word Cloud")

# # Generate the word cloud image
# wordcloud = w.WordCloud(collocations=False, background_color='white', width=3000, height=2000).generate_from_frequencies(d)

# # Display the word cloud using st.image
# st.image(wordcloud.to_array(), use_container_width=True)

# # Optionally, you can add some explanation or text
# st.write("Word cloud representing the most used skills.")

# # Display the Streamlit app
# st.pyplot()

def recommend_jobs(user_skills):
    pred = vectorizer.transform([user_skills.lower()])
    output = svm.predict(pred)[0]
    return output, df_job[df_job['Label'] == output]

# # Streamlit app
st.title("Job Recommendation System")

user_input = st.text_input("Enter your skills (comma-separated)", "")

if st.button("Recommend Jobs"):
    if user_input:
        output, labelData = recommend_jobs(user_input)
        st.write(f"You may look into {output} jobs")
        st.write(f"Here is a list of jobs that is under {output}")
        cos = []
        for index, row in labelData.iterrows():
            skills = [row['skills']]
            skillVec = vectorizer.transform(skills)
            cos_lib = cosine_similarity(skillVec, svm_predictions)
            cos.append(cos_lib[0][0])
        labelData['cosine_similarity'] = cos
        top_5 = labelData.sort_values('cosine_similarity', ascending=False)[
            ['advertiserurl', 'company', 'employmenttype_jobstatus', 'jobdescription', 'joblocation_address', 'jobtitle', 'skills', 'Label']
        ]
        st.subheader("Top 5 Recommendations:")
        st.dataframe(top_5.head(5))
    else:
        st.warning("Please enter your skills.")